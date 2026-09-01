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

        skill_candidates = (
            "labels/filtered_skill_id",
            "labels/predicted_skill_id",
            "labels/skill_id",
        )
        skill_labels = next(
            (name for name in skill_candidates if all(name in demo for demo in demos)),
            None,
        )
        return H5Summary(
            path=input_path,
            layout=layout,
            samples=sum(int(demo["features"].shape[0]) for demo in demos if "features" in demo),
            episodes=len(demos),
            features=int(first_features.shape[1]),
            feature_names=decode_feature_names(first_features),
            skill_labels=skill_labels,
            schema_version=schema_version,
        )


def _copy_attributes(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for name, value in source.items():
        try:
            target[name] = value
        except TypeError:
            target[name] = str(value)


def _convert_flat(source: h5py.File, output: h5py.File) -> None:
    source_features = source["features"]
    source_episodes = source.get("episode_ids", source.get("episodes", source.get("labels")))
    if not isinstance(source_episodes, h5py.Dataset):
        raise ValueError("Flat input is missing episode IDs.")
    original_episode_ids = source_episodes[:]
    if not np.isfinite(original_episode_ids).all() or np.any(original_episode_ids < 0):
        raise ValueError("Flat input episode IDs must be finite and non-negative.")
    boundaries = np.flatnonzero(np.r_[True, original_episode_ids[1:] != original_episode_ids[:-1], True])
    data = output.create_group("data")
    for episode_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        demo = data.create_group(f"demo_{episode_index:06d}")
        features = demo.create_dataset(
            "features",
            data=source_features[start:end],
            chunks=True,
            compression="gzip",
        )
        _copy_attributes(source_features.attrs, features.attrs)
        demo.create_dataset(
            "timestamps/sim",
            data=source["timestamps"][start:end],
            chunks=True,
            compression="gzip",
        )
        demo.create_dataset(
            "episode/index",
            data=np.full(end - start, episode_index, dtype=np.int64),
            chunks=True,
            compression="gzip",
        )
        demo.create_dataset(
            "labels/skill_id",
            data=np.full(end - start, -1, dtype=np.int64),
            chunks=True,
            compression="gzip",
        )

    source_group = output.create_group("source")
    main_names = {"features", "timestamps", "episode_ids", "episodes", "labels"}
    for name in source.keys():
        if name not in main_names:
            source.copy(name, source_group, name=name)


def _convert_multi_episode(source: h5py.File, output: h5py.File) -> None:
    source.copy("data", output)
    for name in source.keys():
        if name != "data" and name not in output:
            source.copy(name, output, name=name)


def convert_h5(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert a supported input to the canonical episode-grouped layout.

    The input file is never modified. Already grouped canonical input is copied
    without flattening or discarding labels and metadata.
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
