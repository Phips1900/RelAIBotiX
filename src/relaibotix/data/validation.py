"""Validation for flat and multi-episode RelAIBotiX HDF5 inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from .h5 import H5Summary, decode_feature_names, detect_layout, inspect_h5


IssueLevel = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    level: IssueLevel
    code: str
    message: str
    location: str = "/"


@dataclass(frozen=True)
class ValidationReport:
    path: Path
    summary: H5Summary | None
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level == "warning")


def _issue(
    issues: list[ValidationIssue],
    level: IssueLevel,
    code: str,
    message: str,
    location: str = "/",
) -> None:
    issues.append(ValidationIssue(level, code, message, location))


def _validate_feature_dataset(
    dataset: h5py.Dataset,
    issues: list[ValidationIssue],
    location: str,
    expected_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if dataset.ndim != 2:
        _issue(issues, "error", "features.ndim", "Features must be two-dimensional.", location)
        return ()

    names = decode_feature_names(dataset)
    if not names:
        _issue(issues, "error", "features.names_missing", "Feature names are required.", location)
    elif len(names) != dataset.shape[1]:
        _issue(
            issues,
            "error",
            "features.names_length",
            f"Found {len(names)} feature names for {dataset.shape[1]} columns.",
            location,
        )
    elif len(set(names)) != len(names):
        _issue(issues, "error", "features.names_duplicate", "Feature names must be unique.", location)

    if expected_names is not None and names != expected_names:
        _issue(
            issues,
            "error",
            "features.inconsistent",
            "Feature names or ordering differ between episodes.",
            location,
        )

    row_step = max(1, min(100_000, dataset.shape[0]))
    for start in range(0, dataset.shape[0], row_step):
        if not np.isfinite(dataset[start : start + row_step]).all():
            _issue(
                issues,
                "error",
                "features.non_finite",
                "Features contain NaN or infinite values.",
                location,
            )
            break
    return names


def _validate_flat(h5_file: h5py.File, issues: list[ValidationIssue]) -> None:
    features = h5_file.get("features")
    if not isinstance(features, h5py.Dataset):
        _issue(issues, "error", "features.missing", "Missing root 'features' dataset.")
        return
    _validate_feature_dataset(features, issues, "/features")
    sample_count = int(features.shape[0]) if features.ndim else 0

    required_datasets = {
        "timestamps": h5_file.get("timestamps"),
        "episode_ids": h5_file.get(
            "episode_ids",
            h5_file.get("episodes", h5_file.get("labels")),
        ),
    }
    for name, dataset in required_datasets.items():
        if not isinstance(dataset, h5py.Dataset):
            _issue(issues, "error", f"{name}.missing", f"Missing root '{name}' dataset.")
        elif dataset.ndim != 1 or dataset.shape[0] != sample_count:
            _issue(
                issues,
                "error",
                f"{name}.shape",
                f"'{name}' must have shape ({sample_count},).",
                f"/{dataset.name.lstrip('/')}",
            )

    for name in ("predicted_labels", "labels_pred", "skills/predicted"):
        if name in h5_file:
            dataset = h5_file[name]
            if dataset.ndim != 1 or dataset.shape[0] != sample_count:
                _issue(
                    issues,
                    "error",
                    "skills.shape",
                    f"'{name}' must have shape ({sample_count},).",
                    f"/{name}",
                )
            break
    else:
        _issue(
            issues,
            "warning",
            "skills.not_run",
            "No skill predictions are present; mandatory skill inference must run before analysis.",
        )


def _validate_multi_episode(h5_file: h5py.File, issues: list[ValidationIssue]) -> None:
    data_group = h5_file.get("data")
    if not isinstance(data_group, h5py.Group) or len(data_group) == 0:
        _issue(issues, "error", "episodes.missing", "The root 'data' group has no episodes.")
        return

    expected_names: tuple[str, ...] | None = None
    skill_values: set[int] = set()
    for demo_name in sorted(data_group.keys()):
        demo = data_group[demo_name]
        if not isinstance(demo, h5py.Group):
            _issue(issues, "error", "episode.not_group", "Episode entry must be a group.", f"/data/{demo_name}")
            continue
        features = demo.get("features")
        if not isinstance(features, h5py.Dataset):
            _issue(issues, "error", "features.missing", "Episode is missing 'features'.", f"/data/{demo_name}")
            continue
        names = _validate_feature_dataset(
            features,
            issues,
            f"/data/{demo_name}/features",
            expected_names,
        )
        if expected_names is None:
            expected_names = names
        sample_count = int(features.shape[0]) if features.ndim else 0

        required = ("timestamps/sim", "episode/index")
        for relative_path in required:
            dataset = demo.get(relative_path)
            if not isinstance(dataset, h5py.Dataset):
                _issue(
                    issues,
                    "error",
                    "episode.dataset_missing",
                    f"Episode is missing '{relative_path}'.",
                    f"/data/{demo_name}",
                )
            elif dataset.ndim != 1 or dataset.shape[0] != sample_count:
                _issue(
                    issues,
                    "error",
                    "episode.dataset_shape",
                    f"'{relative_path}' must have shape ({sample_count},).",
                    f"/data/{demo_name}/{relative_path}",
                )

        skill_ids = demo.get("labels/skill_id")
        if isinstance(skill_ids, h5py.Dataset):
            if skill_ids.ndim != 1 or skill_ids.shape[0] != sample_count:
                _issue(
                    issues,
                    "error",
                    "skills.shape",
                    f"'labels/skill_id' must have shape ({sample_count},).",
                    f"/data/{demo_name}/labels/skill_id",
                )
            else:
                skill_values.update(int(value) for value in np.unique(skill_ids[:]))

    if not skill_values or skill_values == {-1}:
        _issue(
            issues,
            "warning",
            "skills.not_run",
            "Skill IDs are absent or unlabeled; mandatory skill inference must run before analysis.",
        )


def validate_h5(path: str | Path) -> ValidationReport:
    """Validate an HDF5 input without changing it."""

    input_path = Path(path)
    issues: list[ValidationIssue] = []
    if not input_path.is_file():
        _issue(issues, "error", "file.missing", "Input file does not exist.")
        return ValidationReport(input_path, None, tuple(issues))

    try:
        with h5py.File(input_path, "r") as h5_file:
            layout = detect_layout(h5_file)
            if layout == "flat":
                _validate_flat(h5_file, issues)
            else:
                _validate_multi_episode(h5_file, issues)
        summary = inspect_h5(input_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        _issue(issues, "error", "file.unsupported", str(error))
        summary = None

    return ValidationReport(input_path, summary, tuple(issues))
