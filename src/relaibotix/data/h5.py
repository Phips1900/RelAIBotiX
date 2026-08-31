"""Read structural information from supported RelAIBotiX HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import h5py


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
            episode_dataset = h5_file.get("episodes", h5_file.get("labels"))
            episode_count = 0
            if isinstance(episode_dataset, h5py.Dataset) and episode_dataset.size:
                episode_ids = episode_dataset[:]
                episode_count = len({int(value) for value in episode_ids if value >= 0})

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
