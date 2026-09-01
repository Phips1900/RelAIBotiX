"""Read structural information from supported RelAIBotiX HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

import h5py
import numpy as np


H5Layout = Literal["flat", "multi_episode"]


@dataclass(frozen=True)
class H5Summary:
    """Small, read-only description of an HDF5 input file."""

    path: Path
    layout: H5Layout
    samples: int
    episodes: int
    features: int
    feature_names: tuple[str, ...]
    skill_labels: str | None
    schema_version: str | None


def decode_feature_names(dataset: h5py.Dataset) -> tuple[str, ...]:
    """Return feature names from either JSON or the normal HDF5 attribute."""

    names_json = dataset.attrs.get("feature_names_json")
    if names_json:
        if isinstance(names_json, bytes):
            names_json = names_json.decode("utf-8")
        return tuple(str(name) for name in json.loads(str(names_json)))

    raw_names = dataset.attrs.get("feature_names", ())
    return tuple(
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in raw_names
    )


def detect_layout(h5_file: h5py.File) -> H5Layout:
    """Detect the two layouts currently used by the case studies."""

    if "features" in h5_file:
        return "flat"
    if "data" in h5_file and isinstance(h5_file["data"], h5py.Group):
        return "multi_episode"
    raise ValueError(
        "Unsupported HDF5 layout: expected a root 'features' dataset or a 'data' group."
    )


def inspect_h5(path: str | Path) -> H5Summary:
    """Inspect an HDF5 file without loading its full feature matrix."""

    input_path = Path(path)
    with h5py.File(input_path, "r") as h5_file:
        layout = detect_layout(h5_file)
        schema_version = h5_file.attrs.get("schema_version")
        if isinstance(schema_version, bytes):
            schema_version = schema_version.decode("utf-8")
        if schema_version is not None:
            schema_version = str(schema_version)

        if layout == "flat":
            features = h5_file["features"]
            if features.ndim != 2:
                raise ValueError("Root 'features' must be a two-dimensional dataset.")
            episode_dataset = h5_file.get(
                "episode_ids",
                h5_file.get("episodes", h5_file.get("labels")),
            )
            episode_count = 0
            if isinstance(episode_dataset, h5py.Dataset) and episode_dataset.size:
                episode_ids = episode_dataset[:]
                valid = np.isfinite(episode_ids) & (episode_ids >= 0)
                episode_count = int(
                    valid[0]
                    + np.sum(valid[1:] & ((episode_ids[1:] != episode_ids[:-1]) | ~valid[:-1]))
                )

            skill_labels = next(
                (
                    name
                    for name in ("skills/predicted", "predicted_labels", "labels_pred")
                    if name in h5_file
                ),
                None,
            )
            return H5Summary(
                path=input_path,
                layout=layout,
                samples=int(features.shape[0]),
                episodes=episode_count,
                features=int(features.shape[1]),
                feature_names=decode_feature_names(features),
                skill_labels=skill_labels,
                schema_version=schema_version,
            )

        demos = [
            h5_file["data"][name]
            for name in sorted(h5_file["data"].keys())
            if isinstance(h5_file["data"][name], h5py.Group)
        ]
        if not demos:
            raise ValueError("The root 'data' group does not contain any episodes.")
        first_features = demos[0].get("features")
        if not isinstance(first_features, h5py.Dataset) or first_features.ndim != 2:
            raise ValueError("Each episode must contain a two-dimensional 'features' dataset.")

        has_skills = all("labels/skill_id" in demo for demo in demos)
        return H5Summary(
            path=input_path,
            layout=layout,
            samples=sum(int(demo["features"].shape[0]) for demo in demos if "features" in demo),
            episodes=len(demos),
            features=int(first_features.shape[1]),
            feature_names=decode_feature_names(first_features),
            skill_labels="labels/skill_id" if has_skills else None,
            schema_version=schema_version,
        )


def _copy_attributes(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for name, value in source.items():
        try:
            target[name] = value
        except TypeError:
            target[name] = str(value)


def _copy_rows(source: h5py.Dataset, target: h5py.Dataset, row_count: int) -> None:
    step = max(1, min(100_000, row_count))
    for start in range(0, row_count, step):
        target[start : start + step] = source[start : start + step]


def _create_canonical_datasets(
    output: h5py.File,
    sample_count: int,
    feature_count: int,
    feature_dtype: np.dtype,
) -> tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset]:
    features = output.create_dataset(
        "features",
        shape=(sample_count, feature_count),
        dtype=feature_dtype,
        chunks=True,
        compression="gzip",
    )
    timestamps = output.create_dataset(
        "timestamps",
        shape=(sample_count,),
        dtype=np.float64,
        chunks=True,
        compression="gzip",
    )
    episode_ids = output.create_dataset(
        "episode_ids",
        shape=(sample_count,),
        dtype=np.int64,
        chunks=True,
        compression="gzip",
    )
    return features, timestamps, episode_ids


def _convert_flat(source: h5py.File, output: h5py.File) -> None:
    source_features = source["features"]
    sample_count, feature_count = source_features.shape
    features, timestamps, episode_ids = _create_canonical_datasets(
        output,
        int(sample_count),
        int(feature_count),
        source_features.dtype,
    )
    _copy_rows(source_features, features, int(sample_count))
    _copy_rows(source["timestamps"], timestamps, int(sample_count))
    source_episodes = source.get("episode_ids", source.get("episodes", source.get("labels")))
    if not isinstance(source_episodes, h5py.Dataset):
        raise ValueError("Flat input is missing episode IDs.")
    original_episode_ids = source_episodes[:]
    if not np.isfinite(original_episode_ids).all() or np.any(original_episode_ids < 0):
        raise ValueError("Flat input episode IDs must be finite and non-negative.")
    boundaries = np.r_[True, original_episode_ids[1:] != original_episode_ids[:-1]]
    episode_ids[:] = np.cumsum(boundaries, dtype=np.int64) - 1
    _copy_attributes(source_features.attrs, features.attrs)

    source_group = output.create_group("source")
    main_names = {"features", "timestamps", "episode_ids", "episodes", "labels"}
    for name in source.keys():
        if name not in main_names:
            source.copy(name, source_group, name=name)


def _convert_multi_episode(source: h5py.File, output: h5py.File) -> None:
    demos = [source["data"][name] for name in sorted(source["data"].keys())]
    first_features = demos[0]["features"]
    sample_count = sum(int(demo["features"].shape[0]) for demo in demos)
    feature_count = int(first_features.shape[1])
    features, timestamps, episode_ids = _create_canonical_datasets(
        output,
        sample_count,
        feature_count,
        first_features.dtype,
    )
    _copy_attributes(first_features.attrs, features.attrs)

    has_source_skills = all("labels/skill_id" in demo for demo in demos)
    source_skills = None
    if has_source_skills:
        source_group = output.create_group("source")
        source_skills = source_group.create_dataset(
            "skill_ids",
            shape=(sample_count,),
            dtype=np.int64,
            chunks=True,
            compression="gzip",
        )

    offset = 0
    for episode_index, demo in enumerate(demos):
        length = int(demo["features"].shape[0])
        end = offset + length
        features[offset:end] = demo["features"][:]
        timestamps[offset:end] = demo["timestamps/sim"][:]
        episode_ids[offset:end] = episode_index
        if source_skills is not None:
            source_skills[offset:end] = demo["labels/skill_id"][:]
        offset = end


def convert_h5(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert a supported input to the canonical, flat RelAIBotiX layout.

    The input file is never modified. Existing skill predictions are retained below
    ``/source`` and are not treated as fresh detector output.
    """

    source_path = Path(input_path)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()

    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as output:
            layout = detect_layout(source)
            output.attrs["format"] = "relaibotix_hdf5"
            output.attrs["schema_version"] = "1.0"
            output.attrs["source_layout"] = layout
            output.attrs["source_filename"] = source_path.name
            metadata = output.create_group("source_metadata")
            _copy_attributes(source.attrs, metadata.attrs)
            if layout == "flat":
                _convert_flat(source, output)
            else:
                _convert_multi_episode(source, output)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination
