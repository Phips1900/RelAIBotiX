"""Reproducible multi-experiment execution and publication-table export."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from .behavioral import BehavioralAnalyzer, BehavioralThresholds
from .reliability import RobotConfig


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    setting: str
    platform: str
    task: str
    policy: str
    input_h5: Path
    robot_config: Path
    scope: str
    skill_names: Mapping[int, str]
    label_source: str


@dataclass(frozen=True)
class ExperimentManifest:
    name: str
    path: Path
    experiments: tuple[ExperimentSpec, ...]


def _resolve(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Experiment field '{field}' must be a non-empty path.")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    """Load the small, versioned publication experiment schema."""

    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get("schema_version") != "1.0":
        raise ValueError("Experiment manifest schema_version must be '1.0'.")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Experiment manifest requires a non-empty name.")
    entries = raw.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Experiment manifest requires a non-empty experiments list.")

    required = {
        "id", "setting", "platform", "task", "policy", "input_h5",
        "robot_config", "scope", "skill_names", "label_source",
    }
    experiments: list[ExperimentSpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Experiment {index} must be an object.")
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError(f"Experiment {index} is missing fields: {', '.join(missing)}")
        if entry["label_source"] != "existing_detector_predictions":
            raise ValueError(
                "Publication manifests currently require label_source "
                "'existing_detector_predictions'."
            )
        raw_names = entry["skill_names"]
        if not isinstance(raw_names, dict) or not raw_names:
            raise ValueError(f"Experiment {index} skill_names must be a non-empty object.")
        experiments.append(ExperimentSpec(
            experiment_id=str(entry["id"]),
            setting=str(entry["setting"]),
            platform=str(entry["platform"]),
            task=str(entry["task"]),
            policy=str(entry["policy"]),
            input_h5=_resolve(manifest_path.parent, entry["input_h5"], "input_h5"),
            robot_config=_resolve(
                manifest_path.parent, entry["robot_config"], "robot_config"
            ),
            scope=str(entry["scope"]),
            skill_names={int(key): str(value) for key, value in raw_names.items()},
            label_source=str(entry["label_source"]),
        ))
    identifiers = [experiment.experiment_id for experiment in experiments]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Experiment IDs must be unique.")
    return ExperimentManifest(name, manifest_path, tuple(experiments))


def configured_analyzer(
    config: RobotConfig,
    skill_names: Mapping[int, str],
) -> BehavioralAnalyzer:
    assumptions = config.exposure_assumptions
    return BehavioralAnalyzer(
        thresholds=BehavioralThresholds(
            position_step=assumptions.position_step,
            velocity_active=assumptions.velocity_active,
            effort_active=assumptions.effort_active,
            velocity_bands=assumptions.velocity_bands,
            effort_bands=assumptions.effort_bands,
        ),
        component_features={
            name: component.features
            for name, component in config.components.items()
            if any(component.features.values())
        },
        skill_names=skill_names,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latex(value: object) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_")


def write_experiment_summary(
    manifest: ExperimentManifest,
    output_directory: str | Path,
    *,
    solver_metadata: Mapping[str, object],
) -> tuple[Path, Path, Path, Path]:
    """Combine completed experiment outputs into CSV, Markdown, LaTeX, and provenance."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for experiment in manifest.experiments:
        experiment_output = output / experiment.experiment_id
        behavior = json.loads((experiment_output / "behavior" / "behavior.json").read_text())
        reliability = json.loads(
            (experiment_output / "reliability" / "reliability.json").read_text()
        )
        sensitivity = pd.read_csv(experiment_output / "reliability" / "sensitivity.csv")
        episode_keys = {str(row["episode_key"]) for row in behavior["segments"]}
        runs = len(episode_keys)
        total_time = sum(float(row["duration"]) for row in behavior["segments"])
        joint_travel = sum(
            float(row["total_traveled_distance"])
            for row in behavior["joint_summary"]
            if str(row["joint"]) != "gripper"
        )
        critical = sensitivity.head(2)["component"].astype(str).tolist()
        rows.append({
            "setting": experiment.setting,
            "platform": experiment.platform,
            "task": experiment.task,
            "policy": experiment.policy,
            "scope": experiment.scope,
            "runs": runs,
            "average_time_per_run_s": total_time / runs,
            "average_cumulative_joint_travel_per_run_rad": joint_travel / runs,
            "system_failure_probability_per_run": reliability["dtmc"]["failure_probability"],
            "mttf_h": reliability["repeated_run_mttf"]["hours"],
            "critical_components": " & ".join(critical),
        })
        provenance.append({
            "experiment_id": experiment.experiment_id,
            "input_h5": str(experiment.input_h5),
            "input_sha256": sha256_file(experiment.input_h5),
            "robot_config": str(experiment.robot_config),
            "robot_config_sha256": sha256_file(experiment.robot_config),
            "label_source": experiment.label_source,
            "skill_names": dict(experiment.skill_names),
            "runs": runs,
        })

    csv_path = output / "paper_results.csv"
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = (
        "Setting", "Task", "Policy", "Runs", "Avg. time/run (s)",
        "Joint travel/run (rad)", "Failure probability/run", "MTTF (h)",
        "Critical components",
    )
    markdown = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        markdown.append(
            "| " + " | ".join((
                str(row["setting"]), str(row["task"]), str(row["policy"]),
                str(row["runs"]), f'{row["average_time_per_run_s"]:.2f}',
                f'{row["average_cumulative_joint_travel_per_run_rad"]:.2f}',
                f'{row["system_failure_probability_per_run"]:.3e}',
                f'{row["mttf_h"]:.0f}', str(row["critical_components"]),
            )) + " |"
        )
    markdown_path = output / "paper_results.md"
    markdown_path.write_text("\n".join(markdown) + "\n")

    latex_lines = [
        r"\begin{tabular}{lllrrrrrl}",
        r"\toprule",
        r"Setting & Task & Policy & Runs & Time/run (s) & Travel/run (rad) & $P_f$/run & MTTF (h) & Critical components \\",
        r"\midrule",
    ]
    for row in rows:
        latex_lines.append(
            f'{_latex(row["setting"])} & {_latex(row["task"])} & {_latex(row["policy"])} & '
            f'{row["runs"]} & {row["average_time_per_run_s"]:.2f} & '
            f'{row["average_cumulative_joint_travel_per_run_rad"]:.2f} & '
            f'{row["system_failure_probability_per_run"]:.3e} & '
            f'{row["mttf_h"]:.0f} & '
            f'{_latex(row["critical_components"])} ' + r"\\"
        )
    latex_lines.extend((r"\bottomrule", r"\end{tabular}"))
    latex_path = output / "paper_results.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n")

    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps({
        "schema_version": "1.0",
        "manifest": str(manifest.path),
        "manifest_sha256": sha256_file(manifest.path),
        "method": {
            "interval_attribution": "left_endpoint",
            "dtmc_termination": "each_contiguous_recording_transitions_to_done",
            "mttf": "completed_runs_restart; failures_are_absorbing",
            "sensitivity_factor": 10.0,
        },
        "solvers": dict(solver_metadata),
        "experiments": provenance,
    }, indent=2) + "\n")
    return csv_path, markdown_path, latex_path, provenance_path
