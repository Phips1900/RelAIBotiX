"""Case-study-independent behavioral analysis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from relaibotix.data.h5 import decode_feature_names

from .results import BehavioralResult


@dataclass(frozen=True)
class BehavioralThresholds:
    """Activity thresholds and exposure-band boundaries in signal-native units."""

    position_step: float = 1e-3
    velocity_active: float = 3e-2
    effort_active: float = 1e-1
    velocity_bands: tuple[float, float] = (0.5, 1.0)
    effort_bands: tuple[float, float] = (0.2, 0.6)


def _joint_features(feature_names: Sequence[str]) -> dict[str, dict[str, int]]:
    joints: dict[str, dict[str, int]] = {}
    pattern = re.compile(r"^joint_(pos|vel|eff|effort|torque|tau)_(.+)$", re.IGNORECASE)
    for index, feature_name in enumerate(feature_names):
        match = pattern.match(str(feature_name))
        if match is None:
            continue
        signal, identifier = match.groups()
        signal = "effort" if signal.lower() in {"eff", "effort", "torque", "tau"} else signal.lower()
        joint = f"j{int(identifier)}" if identifier.isdigit() else identifier
        joints.setdefault(joint, {})[signal] = index

    for index, feature_name in enumerate(feature_names):
        signal = str(feature_name).lower()
        if signal == "gripper_state":
            joints.setdefault("gripper", {})["pos"] = index
        elif signal in {"gripper_effort", "gripper_torque"}:
            joints.setdefault("gripper", {})["effort"] = index

    actuator_pattern = re.compile(r"^actuator_(?:force|effort|torque)_(.+)$", re.IGNORECASE)
    for index, feature_name in enumerate(feature_names):
        match = actuator_pattern.match(str(feature_name))
        if match is None:
            continue
        identifier = match.group(1).lower()
        candidate = f"joint_{identifier.removesuffix('_vel')}"
        joint = candidate if candidate in joints else f"actuator_{identifier}"
        joints.setdefault(joint, {})["effort"] = index
    return joints


def _finite_stat(values: np.ndarray, operation) -> float:
    finite = values[np.isfinite(values)]
    return float(operation(finite)) if finite.size else float("nan")


def _exposure(values: np.ndarray | None, dt: np.ndarray, active: float, bands: tuple[float, float]) -> tuple[float, float, float, np.ndarray]:
    if values is None or values.size <= 1:
        return 0.0, 0.0, 0.0, np.zeros(dt.size, dtype=bool)
    magnitude = np.abs(values[:-1])
    finite = np.isfinite(magnitude) & np.isfinite(dt) & (dt >= 0.0)
    active_mask = finite & (magnitude > active)
    low = active_mask & (magnitude <= bands[0])
    medium = active_mask & (magnitude > bands[0]) & (magnitude <= bands[1])
    high = active_mask & (magnitude > bands[1])
    return (
        float(np.sum(dt[low])),
        float(np.sum(dt[medium])),
        float(np.sum(dt[high])),
        active_mask,
    )


class BehavioralAnalyzer:
    """Calculate time, position, velocity, and effort metrics from labeled trajectories."""

    def __init__(
        self,
        *,
        thresholds: BehavioralThresholds | None = None,
        skill_names: Mapping[int, str] | None = None,
    ) -> None:
        self.thresholds = thresholds or BehavioralThresholds()
        self.skill_names = dict(skill_names or {})

    def analyze(
        self,
        *,
        features: np.ndarray | pd.DataFrame,
        feature_names: Sequence[str],
        skill_labels: np.ndarray,
        timestamps: np.ndarray,
        episode_ids: np.ndarray,
        episode_keys: np.ndarray | None = None,
        skill_names: Mapping[int, str] | None = None,
    ) -> BehavioralResult:
        values = np.asarray(features, dtype=float)
        labels = np.asarray(skill_labels).reshape(-1)
        times = np.asarray(timestamps, dtype=float).reshape(-1)
        episodes = np.asarray(episode_ids).reshape(-1)
        keys = (
            np.asarray(episode_keys, dtype=object).reshape(-1)
            if episode_keys is not None
            else episodes.astype(str)
        )
        self._validate(values, feature_names, labels, times, episodes)
        if keys.size != len(values):
            raise ValueError("Episode keys must have the same length as features.")
        names = dict(skill_names or {})
        names.update(self.skill_names)

        joints = _joint_features(feature_names)
        if not joints:
            raise ValueError("No joint position, velocity, or effort features were found.")

        segment_rows: list[dict[str, object]] = []
        joint_rows: list[dict[str, object]] = []
        for segment_index, (start, end) in enumerate(self._segments(labels, episodes)):
            episode_id = int(episodes[start])
            skill_id = int(labels[start])
            skill = names.get(skill_id, str(skill_id))
            segment_times = times[start : end + 1]
            duration = float(segment_times[-1] - segment_times[0]) if end > start else 0.0
            segment = {
                "episode_id": episode_id,
                "episode_key": str(keys[start]),
                "segment_index": segment_index,
                "skill_id": skill_id,
                "skill": skill,
                "start_index": start,
                "end_index": end,
                "start_time": float(segment_times[0]),
                "end_time": float(segment_times[-1]),
                "duration": duration,
                "samples": end - start + 1,
            }
            segment_rows.append(segment)
            for joint, columns in joints.items():
                joint_rows.append(
                    self._joint_metrics(values[start : end + 1], segment_times, columns, joint, segment)
                )

        segments = pd.DataFrame(segment_rows)
        joint_metrics = pd.DataFrame(joint_rows)
        return BehavioralResult(
            segments=segments,
            joint_metrics=joint_metrics,
            skill_summary=self._summarize_skills(segments),
            joint_summary=self._summarize_joints(joint_metrics),
        )

    def analyze_h5(
        self,
        input_path: str | Path,
        *,
        skill_labels_dataset: str | None = None,
    ) -> BehavioralResult:
        """Analyze flat legacy input or canonical grouped detector output."""

        with h5py.File(input_path, "r") as source:
            if "data" in source:
                return self._analyze_grouped_h5(source, skill_labels_dataset)

            label_path = skill_labels_dataset or next(
                (
                    name
                    for name in ("skills/predicted", "predicted_labels", "labels_pred")
                    if name in source
                ),
                "skills/predicted",
            )
            episode_path = next(
                (name for name in ("episode_ids", "episodes", "labels") if name in source),
                "episode_ids",
            )
            required = ("features", "timestamps", episode_path, label_path)
            missing = [name for name in required if name not in source]
            if missing:
                raise ValueError(f"HDF5 input is missing required datasets: {', '.join(missing)}")
            feature_dataset = source["features"]
            return self.analyze(
                features=feature_dataset[:],
                feature_names=decode_feature_names(feature_dataset),
                skill_labels=source[label_path][:],
                timestamps=source["timestamps"][:],
                episode_ids=source[episode_path][:],
            )

    def _analyze_grouped_h5(
        self,
        source: h5py.File,
        skill_labels_dataset: str | None,
    ) -> BehavioralResult:
        data = source["data"]
        episode_names = [
            name for name in sorted(data) if isinstance(data[name], h5py.Group) and "features" in data[name]
        ]
        if not episode_names:
            raise ValueError("Canonical HDF5 input contains no '/data/demo_*' episodes.")

        features: list[np.ndarray] = []
        timestamps: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        episode_ids: list[np.ndarray] = []
        episode_keys: list[np.ndarray] = []
        reference_names: tuple[str, ...] | None = None
        detected_skill_names: dict[int, str] = {}
        prediction_paths = (
            (skill_labels_dataset.lstrip("/"),)
            if skill_labels_dataset
            else (
                "labels/filtered_skill_id",
                "labels/predicted_skill_id",
                "labels/skill_id",
            )
        )

        for episode_index, episode_name in enumerate(episode_names):
            episode = data[episode_name]
            feature_dataset = episode["features"]
            names = decode_feature_names(feature_dataset)
            if reference_names is None:
                reference_names = names
            elif names != reference_names:
                raise ValueError(f"/data/{episode_name} uses a different feature schema.")
            label_path = next((path for path in prediction_paths if path in episode), None)
            if label_path is None:
                raise ValueError(f"/data/{episode_name} has no usable skill-label dataset.")
            if "timestamps/sim" not in episode:
                raise ValueError(f"/data/{episode_name} is missing timestamps/sim.")

            episode_features = np.asarray(feature_dataset, dtype=float)
            episode_times = np.asarray(episode["timestamps/sim"], dtype=float).reshape(-1)
            episode_labels = np.asarray(episode[label_path], dtype=np.int64).reshape(-1)
            sample_count = len(episode_features)
            if episode_times.size != sample_count or episode_labels.size != sample_count:
                raise ValueError(f"/data/{episode_name} features, timestamps, and labels are not aligned.")

            label_dataset = episode[label_path]
            class_ids = label_dataset.attrs.get("class_skill_ids")
            class_names_json = label_dataset.attrs.get("class_names_json")
            if class_ids is not None and class_names_json is not None:
                if isinstance(class_names_json, bytes):
                    class_names_json = class_names_json.decode("utf-8")
                class_names = json.loads(str(class_names_json))
                detected_skill_names.update(
                    {int(skill_id): str(name) for skill_id, name in zip(class_ids, class_names)}
                )

            features.append(episode_features)
            timestamps.append(episode_times)
            labels.append(episode_labels)
            episode_ids.append(np.full(sample_count, episode_index, dtype=np.int64))
            episode_keys.append(np.full(sample_count, episode_name, dtype=object))

        assert reference_names is not None
        return self.analyze(
            features=np.concatenate(features),
            feature_names=reference_names,
            skill_labels=np.concatenate(labels),
            timestamps=np.concatenate(timestamps),
            episode_ids=np.concatenate(episode_ids),
            episode_keys=np.concatenate(episode_keys),
            skill_names=detected_skill_names,
        )

    @staticmethod
    def _validate(
        features: np.ndarray,
        feature_names: Sequence[str],
        labels: np.ndarray,
        timestamps: np.ndarray,
        episodes: np.ndarray,
    ) -> None:
        if features.ndim != 2:
            raise ValueError("Features must be a two-dimensional array.")
        sample_count, feature_count = features.shape
        if len(feature_names) != feature_count:
            raise ValueError("Feature-name count does not match the feature matrix.")
        if any(array.size != sample_count for array in (labels, timestamps, episodes)):
            raise ValueError("Features, skill labels, timestamps, and episode IDs must have equal lengths.")
        if sample_count == 0:
            raise ValueError("Behavioral analysis requires at least one sample.")
        if not np.isfinite(labels).all() or np.any(labels < 0):
            raise ValueError("Skill labels must be detector-produced, finite, non-negative IDs.")
        if not np.equal(labels, np.floor(labels)).all():
            raise ValueError("Skill labels must be integer IDs.")
        if not np.isfinite(timestamps).all():
            raise ValueError("Timestamps must be finite.")
        if not np.isfinite(episodes).all() or np.any(episodes < 0):
            raise ValueError("Episode IDs must be finite and non-negative.")
        if not np.equal(episodes, np.floor(episodes)).all():
            raise ValueError("Episode IDs must be integers.")

    @staticmethod
    def _segments(labels: np.ndarray, episodes: np.ndarray) -> list[tuple[int, int]]:
        changes = np.flatnonzero((labels[1:] != labels[:-1]) | (episodes[1:] != episodes[:-1])) + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes - 1, labels.size - 1]
        return [(int(start), int(end)) for start, end in zip(starts, ends)]

    def _joint_metrics(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        columns: Mapping[str, int],
        joint: str,
        segment: Mapping[str, object],
    ) -> dict[str, object]:
        position = features[:, columns["pos"]] if "pos" in columns else None
        velocity = features[:, columns["vel"]] if "vel" in columns else None
        effort = features[:, columns["effort"]] if "effort" in columns else None
        dt = np.diff(timestamps)
        if np.any(dt < 0.0):
            raise ValueError(f"Timestamps decrease inside episode {segment['episode_id']}.")

        traveled_distance = float("nan")
        start_position = end_position = position_range = float("nan")
        position_active = np.zeros(dt.size, dtype=bool)
        if position is not None:
            adjacent = np.isfinite(position[:-1]) & np.isfinite(position[1:])
            steps = np.abs(np.diff(position))
            traveled_distance = float(np.sum(steps[adjacent]))
            position_active = adjacent & (steps > self.thresholds.position_step)
            start_position = _finite_stat(position[:1], np.mean)
            end_position = _finite_stat(position[-1:], np.mean)
            position_range = _finite_stat(position, np.ptp)

        vel_low, vel_medium, vel_high, velocity_active = _exposure(
            velocity, dt, self.thresholds.velocity_active, self.thresholds.velocity_bands
        )
        effort_low, effort_medium, effort_high, effort_active = _exposure(
            effort, dt, self.thresholds.effort_active, self.thresholds.effort_bands
        )
        active_mask = position_active | velocity_active | effort_active
        active_time = float(np.sum(dt[active_mask])) if dt.size else 0.0
        duration = float(segment["duration"])

        return {
            "episode_id": segment["episode_id"],
            "episode_key": segment["episode_key"],
            "segment_index": segment["segment_index"],
            "skill_id": segment["skill_id"],
            "skill": segment["skill"],
            "joint": joint,
            "duration": duration,
            "start_position": start_position,
            "end_position": end_position,
            "position_range": position_range,
            "traveled_distance": traveled_distance,
            "mean_abs_velocity": _finite_stat(np.abs(velocity), np.mean) if velocity is not None else float("nan"),
            "rms_velocity": _finite_stat(velocity, lambda value: np.sqrt(np.mean(value ** 2))) if velocity is not None else float("nan"),
            "max_abs_velocity": _finite_stat(np.abs(velocity), np.max) if velocity is not None else float("nan"),
            "velocity_time_low": vel_low,
            "velocity_time_medium": vel_medium,
            "velocity_time_high": vel_high,
            "mean_abs_effort": _finite_stat(np.abs(effort), np.mean) if effort is not None else float("nan"),
            "rms_effort": _finite_stat(effort, lambda value: np.sqrt(np.mean(value ** 2))) if effort is not None else float("nan"),
            "max_abs_effort": _finite_stat(np.abs(effort), np.max) if effort is not None else float("nan"),
            "effort_time_low": effort_low,
            "effort_time_medium": effort_medium,
            "effort_time_high": effort_high,
            "active_time": active_time,
            "active_fraction": active_time / duration if duration > 0.0 else 0.0,
        }

    @staticmethod
    def _summarize_skills(segments: pd.DataFrame) -> pd.DataFrame:
        return (
            segments.groupby(["skill_id", "skill"], as_index=False)
            .agg(
                n_segments=("segment_index", "count"),
                n_episodes=("episode_key", "nunique"),
                total_duration=("duration", "sum"),
                mean_segment_duration=("duration", "mean"),
                max_segment_duration=("duration", "max"),
            )
        )

    @staticmethod
    def _summarize_joints(metrics: pd.DataFrame) -> pd.DataFrame:
        return (
            metrics.groupby(["skill_id", "skill", "joint"], as_index=False)
            .agg(
                n_segments=("segment_index", "count"),
                total_traveled_distance=("traveled_distance", lambda value: value.sum(min_count=1)),
                mean_traveled_distance=("traveled_distance", "mean"),
                max_traveled_distance=("traveled_distance", "max"),
                total_active_time=("active_time", "sum"),
                mean_active_fraction=("active_fraction", "mean"),
                max_abs_velocity=("max_abs_velocity", "max"),
                mean_rms_velocity=("rms_velocity", "mean"),
                velocity_time_low=("velocity_time_low", "sum"),
                velocity_time_medium=("velocity_time_medium", "sum"),
                velocity_time_high=("velocity_time_high", "sum"),
                max_abs_effort=("max_abs_effort", "max"),
                mean_rms_effort=("rms_effort", "mean"),
                effort_time_low=("effort_time_low", "sum"),
                effort_time_medium=("effort_time_medium", "sum"),
                effort_time_high=("effort_time_high", "sum"),
            )
        )
