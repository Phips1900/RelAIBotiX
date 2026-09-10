"""Create traceable canonical HDF5 subsets without modifying source recordings."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import h5py
import numpy as np

from .h5 import decode_feature_names


def select_h5_episodes(
    input_path: str | Path,
    output_path: str | Path,
    *,
    exclude_episode_keys: Iterable[str] = (),
    drop_feature_names: Iterable[str] = (),
    overwrite: bool = False,
) -> Path:
    """Copy a canonical grouped HDF5 while excluding named episode groups."""

    source_path = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    excluded = tuple(dict.fromkeys(str(key) for key in exclude_episode_keys))
    dropped_features = tuple(dict.fromkeys(str(name) for name in drop_feature_names))
    if source_path == destination:
        raise ValueError("Episode selection must write to a different HDF5 file.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"HDF5 output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary: Path | None = None
    try:
        with h5py.File(source_path, "r") as source:
            if "data" not in source or not isinstance(source["data"], h5py.Group):
                raise ValueError(
                    "Episode selection requires canonical grouped HDF5 input under /data."
                )
            available = tuple(source["data"].keys())
            unknown = sorted(set(excluded) - set(available))
            if unknown:
                raise ValueError(f"Unknown episode keys: {', '.join(unknown)}")
            excluded_set = set(excluded)
            selected = tuple(key for key in available if key not in excluded_set)
            if not selected:
                raise ValueError("Episode selection cannot exclude every episode.")

            handle, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            os.close(handle)
            temporary = Path(temporary_name)
            with h5py.File(temporary, "w") as target:
                for name, value in source.attrs.items():
                    target.attrs[name] = value
                for name in source:
                    if name != "data":
                        source.copy(name, target)
                data = target.create_group("data")
                for key in selected:
                    source.copy(source["data"][key], data, name=key)
                    if dropped_features:
                        source_features = source["data"][key]["features"]
                        names = decode_feature_names(source_features)
                        unknown_features = sorted(set(dropped_features) - set(names))
                        if unknown_features:
                            raise ValueError(
                                f"Unknown feature names: {', '.join(unknown_features)}"
                            )
                        retained_indices = [
                            index for index, name in enumerate(names)
                            if name not in set(dropped_features)
                        ]
                        retained_names = tuple(names[index] for index in retained_indices)
                        copied_episode = data[key]
                        del copied_episode["features"]
                        features = copied_episode.create_dataset(
                            "features",
                            data=source_features[:, retained_indices],
                            chunks=True,
                            compression="gzip",
                        )
                        for name, value in source_features.attrs.items():
                            if name not in {"feature_names", "feature_names_json"}:
                                features.attrs[name] = value
                        if "feature_names_json" in source_features.attrs:
                            features.attrs["feature_names_json"] = json.dumps(retained_names)
                        else:
                            features.attrs["feature_names"] = np.asarray(
                                retained_names, dtype=h5py.string_dtype("utf-8")
                            )
                target.attrs["episode_count"] = len(selected)
                target.attrs["selection_source_hdf5"] = str(source_path)
                target.attrs["selection_excluded_episode_keys_json"] = json.dumps(excluded)
                target.attrs["selection_dropped_feature_names_json"] = json.dumps(
                    dropped_features
                )
                target.attrs["selection_preserves_source_episode_keys"] = True
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
