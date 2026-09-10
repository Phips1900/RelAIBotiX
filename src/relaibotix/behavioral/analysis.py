"""Case-study-independent behavioral analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    """Activity thresholds and exposure-band boundaries.

    Values are interpreted in signal-native units unless a component supplies
    an exposure reference, in which case velocity and effort are fractions of
    that reference.
    """

    position_step: float = 1e-3
    velocity_active: float = 3e-2
    effort_active: float = 1e-1
    velocity_bands: tuple[float, float] = (0.5, 1.0)
    effort_bands: tuple[float, float] = (0.2, 0.6)

    def as_dict(self) -> dict[str, object]:
        return {
            "position_step": self.position_step,
            "velocity_active": self.velocity_active,
            "effort_active": self.effort_active,
            "velocity_bands": list(self.velocity_bands),
            "effort_bands": list(self.effort_bands),
        }


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


def _configured_features(
    feature_names: Sequence[str],
    component_features: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Resolve configured HDF5 columns into reliability-component measurements."""

    indices = {str(name): index for index, name in enumerate(feature_names)}
    signal_names = {
        "position": "pos",
        "velocity": "vel",
        "effort": "effort",
        "torque": "effort",
    }
    resolved: dict[str, dict[str, tuple[int, ...]]] = {}
    for component, signals in component_features.items():
        columns: dict[str, tuple[int, ...]] = {}
        for signal, names in signals.items():
            normalized_signal = signal_names.get(str(signal).lower())
            if normalized_signal is None or not names:
                continue
            missing = [name for name in names if name not in indices]
            if missing:
                raise ValueError(
                    f"Component '{component}' references missing HDF5 features: "
                    + ", ".join(missing)
                )
            columns[normalized_signal] = tuple(indices[name] for name in names)
        if columns:
            resolved[str(component)] = columns
    return resolved


def _mobile_base_features(feature_names: Sequence[str]) -> dict[str, int]:
    indices = {str(name): index for index, name in enumerate(feature_names)}
    return {
        signal: indices[name]
        for signal, name in {
            "x": "base_x_m",
            "y": "base_y_m",
            "yaw": "base_yaw_rad",
            "vx": "base_vx_m_s",
            "vy": "base_vy_m_s",
            "wz": "base_wz_rad_s",
        }.items()
        if name in indices
    }


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


def _continuous_weighted_time(
    values: np.ndarray | None,
    dt: np.ndarray,
    active: float,
    bands: tuple[float, float],
    multipliers: tuple[float, float, float],
) -> float:
    """Integrate a capped piecewise-linear exposure factor over active intervals."""

    if values is None or values.size <= 1:
        return 0.0
    magnitude = np.abs(values[:-1])
    finite = np.isfinite(magnitude) & np.isfinite(dt) & (dt >= 0.0)
    active_mask = finite & (magnitude > active)
    if not np.any(active_mask):
        return 0.0
    factors = np.interp(
        magnitude[active_mask],
        (0.0, bands[0], bands[1]),
        multipliers,
    )
    return float(np.sum(dt[active_mask] * factors))


class BehavioralAnalyzer:
    """Calculate time, position, velocity, and effort metrics from labeled trajectories."""

    def __init__(
        self,
        *,
        thresholds: BehavioralThresholds | None = None,
        skill_names: Mapping[int, str] | None = None,
        component_features: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        component_references: Mapping[str, Mapping[str, float]] | None = None,
        velocity_multipliers: tuple[float, float, float] = (1.0, 1.5, 2.0),
        effort_multipliers: tuple[float, float, float] = (1.0, 2.0, 5.0),
    ) -> None:
        self.thresholds = thresholds or BehavioralThresholds()
        self.skill_names = dict(skill_names or {})
        self.component_features = component_features
        self.component_references = {
            str(component): {str(signal): float(value) for signal, value in references.items()}
            for component, references in (component_references or {}).items()
        }
        self.velocity_multipliers = velocity_multipliers
        self.effort_multipliers = effort_multipliers

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

        joints = (
            _configured_features(feature_names, self.component_features)
            if self.component_features is not None
            else _joint_features(feature_names)
        )
        if not joints:
            raise ValueError("No joint position, velocity, or effort features were found.")
        base_columns = _mobile_base_features(feature_names)

        segment_rows: list[dict[str, object]] = []
        joint_rows: list[dict[str, object]] = []
        base_rows: list[dict[str, object]] = []
        for segment_index, (start, end) in enumerate(self._segments(labels, episodes)):
            episode_id = int(episodes[start])
            skill_id = int(labels[start])
            skill = names.get(skill_id, str(skill_id))
            # Labels use a left-endpoint convention: the label at sample i owns
            # the interval [t_i, t_i+1). Include the first sample of the next
            # skill as an endpoint so no time or motion is lost at a boundary.
            interval_end = (
                end + 1
                if end + 1 < len(values) and episodes[end + 1] == episodes[end]
                else end
            )
            segment_times = times[start : interval_end + 1]
            segment_values = values[start : interval_end + 1]
            duration = (
                float(segment_times[-1] - segment_times[0])
                if segment_times.size > 1
                else 0.0
            )
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
                    self._joint_metrics(
                        segment_values,
                        segment_times,
                        columns,
                        joint,
                        segment,
                        self.component_references.get(joint, {}),
                    )
                )
            if base_columns:
                base_rows.append(
                    self._base_metrics(
                        segment_values, segment_times, base_columns, segment
                    )
                )

        segments = pd.DataFrame(segment_rows)
        joint_metrics = pd.DataFrame(joint_rows)
        base_metrics = pd.DataFrame(base_rows)
        return BehavioralResult(
            segments=segments,
            joint_metrics=joint_metrics,
            skill_summary=self._summarize_skills(segments),
            joint_summary=self._summarize_joints(joint_metrics),
            base_metrics=base_metrics,
            base_summary=self._summarize_base(base_metrics),
            metadata={
                "behavioral_thresholds": self.thresholds.as_dict(),
                "measurement_mapping": (
                    "robot_config" if self.component_features is not None else "feature_names"
                ),
                "interval_attribution": "left_endpoint",
                "component_exposure_references": self.component_references,
                "continuous_exposure_multipliers": {
                    "velocity": list(self.velocity_multipliers),
                    "effort": list(self.effort_multipliers),
                },
            },
        )

    @staticmethod
    def _base_metrics(
        features: np.ndarray,
        timestamps: np.ndarray,
        columns: Mapping[str, int],
        segment: Mapping[str, object],
    ) -> dict[str, object]:
        dt = np.diff(timestamps)
        if np.any(dt < 0.0):
            raise ValueError(f"Timestamps decrease inside episode {segment['episode_id']}.")

        translation_distance = float("nan")
        if "x" in columns and "y" in columns:
            dx = np.diff(features[:, columns["x"]])
            dy = np.diff(features[:, columns["y"]])
            valid = np.isfinite(dx) & np.isfinite(dy)
            translation_distance = float(np.sum(np.hypot(dx[valid], dy[valid])))

        rotation_distance = float("nan")
        if "yaw" in columns:
            delta_yaw = np.diff(features[:, columns["yaw"]])
            wrapped = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
            rotation_distance = float(np.sum(np.abs(wrapped[np.isfinite(wrapped)])))

        owned_samples = int(segment["samples"])
        linear_speed = None
        if "vx" in columns and "vy" in columns:
            linear_speed = np.hypot(
                features[:owned_samples, columns["vx"]],
                features[:owned_samples, columns["vy"]],
            )
        angular_speed = (
            np.abs(features[:owned_samples, columns["wz"]])
            if "wz" in columns
            else None
        )
        return {
            "episode_id": segment["episode_id"],
            "episode_key": segment["episode_key"],
            "segment_index": segment["segment_index"],
            "skill_id": segment["skill_id"],
            "skill": segment["skill"],
            "duration": segment["duration"],
            "translation_distance_m": translation_distance,
            "rotation_distance_rad": rotation_distance,
            "mean_linear_speed_m_s": (
                _finite_stat(linear_speed, np.mean)
                if linear_speed is not None else float("nan")
            ),
            "max_linear_speed_m_s": (
                _finite_stat(linear_speed, np.max)
                if linear_speed is not None else float("nan")
            ),
            "mean_angular_speed_rad_s": (
                _finite_stat(angular_speed, np.mean)
                if angular_speed is not None else float("nan")
            ),
            "max_angular_speed_rad_s": (
                _finite_stat(angular_speed, np.max)
                if angular_speed is not None else float("nan")
            ),
        }

    def analyze_h5(
        self,
        input_path: str | Path,
        *,
        skill_labels_dataset: str | None = None,
        successful_only: bool = False,
        end_after_skill: str | int | None = None,
        exclude_missing_end_skill: bool = False,
        keep_missing_end_skill: bool = False,
    ) -> BehavioralResult:
        """Analyze flat legacy input or canonical grouped detector output."""

        if exclude_missing_end_skill and keep_missing_end_skill:
            raise ValueError(
                "Missing terminal-skill episodes cannot be both excluded and retained."
            )
        with h5py.File(input_path, "r") as source:
            if "data" in source:
                result = self._analyze_grouped_h5(
                    source,
                    skill_labels_dataset,
                    successful_only=successful_only,
                )
                return self._trim_after_final_skill(
                    result,
                    end_after_skill,
                    exclude_missing=exclude_missing_end_skill,
                    keep_missing=keep_missing_end_skill,
                )

            if successful_only:
                raise ValueError(
                    "Successful-only selection requires canonical grouped HDF5 input "
                    "with a task_success attribute on every episode."
                )

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
            timestamps = np.asarray(source["timestamps"][:], dtype=float).reshape(-1)
            source_episode_ids = np.asarray(source[episode_path][:]).reshape(-1)
            if timestamps.size != source_episode_ids.size:
                raise ValueError("Flat HDF5 timestamps and episode IDs are not aligned.")
            boundaries = np.r_[
                True,
                (source_episode_ids[1:] != source_episode_ids[:-1])
                | (timestamps[1:] < timestamps[:-1]),
            ]
            run_ids = np.cumsum(boundaries, dtype=np.int64) - 1
            episode_keys = np.asarray(
                [f"run_{run_id:06d}" for run_id in run_ids],
                dtype=object,
            )
            result = self.analyze(
                features=feature_dataset[:],
                feature_names=decode_feature_names(feature_dataset),
                skill_labels=source[label_path][:],
                timestamps=timestamps,
                episode_ids=run_ids,
                episode_keys=episode_keys,
            )
            return self._trim_after_final_skill(
                result,
                end_after_skill,
                exclude_missing=exclude_missing_end_skill,
                keep_missing=keep_missing_end_skill,
            )

    def _trim_after_final_skill(
        self,
        result: BehavioralResult,
        terminal_skill: str | int | None,
        *,
        exclude_missing: bool,
        keep_missing: bool,
    ) -> BehavioralResult:
        """Retain each episode through its final occurrence of a terminal skill."""

        if terminal_skill is None:
            return result
        segments = result.segments
        if isinstance(terminal_skill, int) or str(terminal_skill).strip().lstrip("-").isdigit():
            terminal_id = int(terminal_skill)
            matches = segments["skill_id"] == terminal_id
            terminal_label = str(terminal_id)
        else:
            terminal_label = str(terminal_skill).strip()
            matches = segments["skill"].str.casefold() == terminal_label.casefold()

        terminal_segments = segments.loc[matches]
        episode_ids = set(map(int, segments["episode_id"].unique()))
        terminal_episode_ids = set(map(int, terminal_segments["episode_id"].unique()))
        missing_episode_ids = sorted(episode_ids - terminal_episode_ids)
        missing_episode_keys = sorted(
            segments.loc[
                segments["episode_id"].isin(missing_episode_ids), "episode_key"
            ].astype(str).unique()
        )
        if missing_episode_ids and not exclude_missing and not keep_missing:
            preview = ", ".join(missing_episode_keys[:5])
            suffix = "..." if len(missing_episode_keys) > 5 else ""
            raise ValueError(
                f"Terminal skill '{terminal_label}' was not detected in "
                f"{len(missing_episode_ids)} episodes: {preview}{suffix}"
            )

        final_segments = terminal_segments.groupby("episode_id")["segment_index"].max()
        keep_episode = (
            pd.Series(True, index=segments.index)
            if keep_missing
            else segments["episode_id"].isin(terminal_episode_ids)
        )
        keep_through_terminal = segments.apply(
            lambda row: (
                keep_missing and int(row["episode_id"]) not in terminal_episode_ids
            ) or (
                int(row["episode_id"]) in terminal_episode_ids
                and int(row["segment_index"])
                <= int(final_segments.loc[int(row["episode_id"])])
            ),
            axis=1,
        )
        kept_segments = segments.loc[keep_episode & keep_through_terminal].copy()
        retained_pairs = pd.MultiIndex.from_frame(
            kept_segments[["episode_id", "segment_index"]]
        )

        def retain_metrics(table: pd.DataFrame) -> pd.DataFrame:
            if table.empty:
                return table.copy()
            pairs = pd.MultiIndex.from_frame(table[["episode_id", "segment_index"]])
            return table.loc[pairs.isin(retained_pairs)].copy()

        joint_metrics = retain_metrics(result.joint_metrics)
        base_metrics = retain_metrics(result.base_metrics)
        return BehavioralResult(
            segments=kept_segments,
            joint_metrics=joint_metrics,
            skill_summary=self._summarize_skills(kept_segments),
            joint_summary=self._summarize_joints(joint_metrics),
            base_metrics=base_metrics,
            base_summary=self._summarize_base(base_metrics),
            metadata={
                **result.metadata,
                "mission_end_policy": "after_final_skill",
                "mission_end_skill": terminal_label,
                "missing_end_skill_policy": (
                    "keep_complete_episode" if keep_missing else
                    "exclude_episode" if exclude_missing else "error"
                ),
                "pre_trim_episode_count": len(episode_ids),
                "analyzed_episode_count": (
                    len(episode_ids) if keep_missing else len(terminal_episode_ids)
                ),
                "excluded_missing_end_skill_episodes": (
                    0 if keep_missing else len(missing_episode_ids)
                ),
                "missing_end_skill_episode_keys": missing_episode_keys,
                "post_terminal_segments_excluded": int(len(segments) - len(kept_segments)),
            },
        )

    def _analyze_grouped_h5(
        self,
        source: h5py.File,
        skill_labels_dataset: str | None,
        *,
        successful_only: bool = False,
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
        excluded_invalid_samples = 0
        analysis_episode_index = 0
        reference_names: tuple[str, ...] | None = None
        detected_skill_names: dict[int, str] = {}
        excluded_unsuccessful_episodes = 0
        prediction_paths = (
            (skill_labels_dataset.lstrip("/"),)
            if skill_labels_dataset
            else (
                "labels/filtered_skill_id",
                "labels/predicted_skill_id",
                "labels/skill_id",
            )
        )

        for episode_name in episode_names:
            episode = data[episode_name]
            if successful_only:
                if "task_success" not in episode.attrs:
                    raise ValueError(
                        f"/data/{episode_name} is missing the task_success attribute "
                        "required for successful-only selection."
                    )
                if not bool(episode.attrs["task_success"]):
                    excluded_unsuccessful_episodes += 1
                    continue
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

            validity = (
                np.asarray(episode["validity/valid"], dtype=bool).reshape(-1)
                if "validity/valid" in episode
                else np.ones(sample_count, dtype=bool)
            )
            if validity.size != sample_count:
                raise ValueError(f"/data/{episode_name}/validity/valid is not aligned with features.")
            excluded_invalid_samples += int(np.count_nonzero(~validity))

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

            starts = np.flatnonzero(validity & ~np.r_[False, validity[:-1]])
            ends = np.flatnonzero(validity & ~np.r_[validity[1:], False]) + 1
            for region_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
                region_key = (
                    episode_name
                    if len(starts) == 1
                    else f"{episode_name}:valid_{region_index:03d}"
                )
                features.append(episode_features[start:end])
                timestamps.append(episode_times[start:end])
                labels.append(episode_labels[start:end])
                episode_ids.append(
                    np.full(end - start, analysis_episode_index, dtype=np.int64)
                )
                episode_keys.append(np.full(end - start, region_key, dtype=object))
                analysis_episode_index += 1

        if not features:
            detail = " successful" if successful_only else " valid"
            raise ValueError(
                f"Canonical HDF5 input contains no{detail} episodes with valid samples to analyze."
            )
        assert reference_names is not None
        result = self.analyze(
            features=np.concatenate(features),
            feature_names=reference_names,
            skill_labels=np.concatenate(labels),
            timestamps=np.concatenate(timestamps),
            episode_ids=np.concatenate(episode_ids),
            episode_keys=np.concatenate(episode_keys),
            skill_names=detected_skill_names,
        )
        return replace(
            result,
            metadata={
                **result.metadata,
                "validity_policy": "exclude_explicitly_invalid_samples",
                "excluded_invalid_samples": excluded_invalid_samples,
                "run_selection": "successful_only" if successful_only else "all_attempts",
                "input_episode_count": len(episode_names),
                "selected_episode_count": len(episode_names) - excluded_unsuccessful_episodes,
                "excluded_unsuccessful_episodes": excluded_unsuccessful_episodes,
            },
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
        columns: Mapping[str, int | tuple[int, ...]],
        joint: str,
        segment: Mapping[str, object],
        references: Mapping[str, float],
    ) -> dict[str, object]:
        def signal_values(signal: str) -> tuple[np.ndarray | None, int]:
            if signal not in columns:
                return None, 0
            raw_columns = columns[signal]
            selected = features[:, raw_columns]
            if selected.ndim == 1:
                return selected, 1
            if selected.shape[1] == 1:
                return selected[:, 0], 1
            # A subsystem's instantaneous load is classified by its most
            # heavily used axis, so elapsed time is counted only once.
            return np.max(np.abs(selected), axis=1), int(selected.shape[1])

        position_columns = columns.get("pos")
        position_matrix = None
        if position_columns is not None:
            position_matrix = features[:, position_columns]
            if position_matrix.ndim == 1:
                position_matrix = position_matrix[:, np.newaxis]
        velocity, velocity_axes = signal_values("vel")
        effort, effort_axes = signal_values("effort")
        owned_samples = int(segment["samples"])
        dt = np.diff(timestamps)
        if np.any(dt < 0.0):
            raise ValueError(f"Timestamps decrease inside episode {segment['episode_id']}.")

        traveled_distance = float("nan")
        start_position = end_position = position_range = float("nan")
        position_active = np.zeros(dt.size, dtype=bool)
        position_axes = 0
        if position_matrix is not None:
            position_axes = int(position_matrix.shape[1])
            adjacent = np.isfinite(position_matrix[:-1]) & np.isfinite(position_matrix[1:])
            steps = np.abs(np.diff(position_matrix, axis=0))
            traveled_distance = float(np.sum(steps[adjacent]))
            position_active = np.any(
                adjacent & (steps > self.thresholds.position_step), axis=1
            )
            if position_axes == 1:
                position = position_matrix[:, 0]
                start_position = _finite_stat(position[:1], np.mean)
                end_position = _finite_stat(position[-1:], np.mean)
                position_range = _finite_stat(position, np.ptp)

        velocity_reference = references.get("velocity")
        effort_reference = references.get("effort")
        distance_reference = references.get("distance")
        velocity_for_exposure = (
            velocity / velocity_reference
            if velocity is not None and velocity_reference is not None
            else velocity
        )
        effort_for_exposure = (
            effort / effort_reference
            if effort is not None and effort_reference is not None
            else effort
        )
        traveled_distance_utilization = (
            traveled_distance / distance_reference
            if distance_reference is not None and np.isfinite(traveled_distance)
            else float("nan")
        )

        vel_low, vel_medium, vel_high, velocity_active = _exposure(
            velocity_for_exposure,
            dt,
            self.thresholds.velocity_active,
            self.thresholds.velocity_bands,
        )
        effort_low, effort_medium, effort_high, effort_active = _exposure(
            effort_for_exposure,
            dt,
            self.thresholds.effort_active,
            self.thresholds.effort_bands,
        )
        velocity_weighted_time_continuous = _continuous_weighted_time(
            velocity_for_exposure,
            dt,
            self.thresholds.velocity_active,
            self.thresholds.velocity_bands,
            self.velocity_multipliers,
        )
        effort_weighted_time_continuous = _continuous_weighted_time(
            effort_for_exposure,
            dt,
            self.thresholds.effort_active,
            self.thresholds.effort_bands,
            self.effort_multipliers,
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
            "position_axes": position_axes,
            "velocity_axes": velocity_axes,
            "effort_axes": effort_axes,
            "duration": duration,
            "start_position": start_position,
            "end_position": end_position,
            "position_range": position_range,
            "traveled_distance": traveled_distance,
            "distance_reference": distance_reference,
            "traveled_distance_utilization": traveled_distance_utilization,
            "mean_abs_velocity": _finite_stat(np.abs(velocity[:owned_samples]), np.mean) if velocity is not None else float("nan"),
            "rms_velocity": _finite_stat(velocity[:owned_samples], lambda value: np.sqrt(np.mean(value ** 2))) if velocity is not None else float("nan"),
            "max_abs_velocity": _finite_stat(np.abs(velocity[:owned_samples]), np.max) if velocity is not None else float("nan"),
            "velocity_reference": velocity_reference,
            "mean_velocity_utilization": _finite_stat(np.abs(velocity_for_exposure[:owned_samples]), np.mean) if velocity_for_exposure is not None and velocity_reference is not None else float("nan"),
            "max_velocity_utilization": _finite_stat(np.abs(velocity_for_exposure[:owned_samples]), np.max) if velocity_for_exposure is not None and velocity_reference is not None else float("nan"),
            "velocity_time_low": vel_low,
            "velocity_time_medium": vel_medium,
            "velocity_time_high": vel_high,
            "velocity_weighted_time_continuous": velocity_weighted_time_continuous,
            "mean_abs_effort": _finite_stat(np.abs(effort[:owned_samples]), np.mean) if effort is not None else float("nan"),
            "rms_effort": _finite_stat(effort[:owned_samples], lambda value: np.sqrt(np.mean(value ** 2))) if effort is not None else float("nan"),
            "max_abs_effort": _finite_stat(np.abs(effort[:owned_samples]), np.max) if effort is not None else float("nan"),
            "effort_reference": effort_reference,
            "mean_effort_utilization": _finite_stat(np.abs(effort_for_exposure[:owned_samples]), np.mean) if effort_for_exposure is not None and effort_reference is not None else float("nan"),
            "max_effort_utilization": _finite_stat(np.abs(effort_for_exposure[:owned_samples]), np.max) if effort_for_exposure is not None and effort_reference is not None else float("nan"),
            "effort_time_low": effort_low,
            "effort_time_medium": effort_medium,
            "effort_time_high": effort_high,
            "effort_weighted_time_continuous": effort_weighted_time_continuous,
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
                total_traveled_distance_utilization=("traveled_distance_utilization", lambda value: value.sum(min_count=1)),
                mean_traveled_distance=("traveled_distance", "mean"),
                mean_traveled_distance_utilization=("traveled_distance_utilization", "mean"),
                max_traveled_distance=("traveled_distance", "max"),
                total_active_time=("active_time", "sum"),
                mean_active_fraction=("active_fraction", "mean"),
                max_abs_velocity=("max_abs_velocity", "max"),
                mean_rms_velocity=("rms_velocity", "mean"),
                mean_velocity_utilization=("mean_velocity_utilization", "mean"),
                max_velocity_utilization=("max_velocity_utilization", "max"),
                velocity_time_low=("velocity_time_low", "sum"),
                velocity_time_medium=("velocity_time_medium", "sum"),
                velocity_time_high=("velocity_time_high", "sum"),
                velocity_weighted_time_continuous=("velocity_weighted_time_continuous", "sum"),
                max_abs_effort=("max_abs_effort", "max"),
                mean_rms_effort=("rms_effort", "mean"),
                mean_effort_utilization=("mean_effort_utilization", "mean"),
                max_effort_utilization=("max_effort_utilization", "max"),
                effort_time_low=("effort_time_low", "sum"),
                effort_time_medium=("effort_time_medium", "sum"),
                effort_time_high=("effort_time_high", "sum"),
                effort_weighted_time_continuous=("effort_weighted_time_continuous", "sum"),
            )
        )

    @staticmethod
    def _summarize_base(metrics: pd.DataFrame) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame()
        return (
            metrics.groupby(["skill_id", "skill"], as_index=False)
            .agg(
                n_segments=("segment_index", "count"),
                total_translation_distance_m=("translation_distance_m", "sum"),
                mean_translation_distance_m=("translation_distance_m", "mean"),
                total_rotation_distance_rad=("rotation_distance_rad", "sum"),
                mean_rotation_distance_rad=("rotation_distance_rad", "mean"),
                max_linear_speed_m_s=("max_linear_speed_m_s", "max"),
                max_angular_speed_rad_s=("max_angular_speed_rad_s", "max"),
            )
        )
