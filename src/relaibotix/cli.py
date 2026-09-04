"""Command-line interface for RelAIBotiX."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from .behavioral import BehavioralAnalyzer
from .behavioral.results import BehavioralResult
from .data import convert_h5, inspect_h5, validate_h5


def _print_summary(path: str | Path) -> None:
    summary = inspect_h5(path)
    print(f"File: {summary.path}")
    print(f"Layout: {summary.layout}")
    print(f"Schema version: {summary.schema_version or 'legacy'}")
    print(f"Episodes: {summary.episodes}")
    print(f"Samples: {summary.samples}")
    print(f"Features: {summary.features}")
    print(f"Skill labels: {summary.skill_labels or 'not available'}")
    print("Feature names:")
    for name in summary.feature_names:
        print(f"  - {name}")


def _print_validation(path: str | Path, config_path: str | Path | None = None) -> bool:
    config = None
    if config_path is not None:
        from .reliability import load_robot_config

        config = load_robot_config(config_path)
    report = validate_h5(path, config=config)
    state = "VALID" if report.valid else "INVALID"
    print(f"{state}: {report.path}")
    if report.summary is not None:
        print(
            f"  {report.summary.layout}, {report.summary.episodes} episodes, "
            f"{report.summary.samples} samples, {report.summary.features} features"
        )
    for issue in report.issues:
        print(f"  {issue.level.upper()} [{issue.code}] {issue.location}: {issue.message}")
    return report.valid


def _h5_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relaibotix h5", description="Inspect, validate, or convert HDF5 input.")
    commands = parser.add_subparsers(dest="h5_command", required=True)

    inspect_parser = commands.add_parser("inspect", help="Show the structure of an HDF5 input.")
    inspect_parser.add_argument("input", type=Path)

    validate_parser = commands.add_parser("validate", help="Validate an HDF5 input without changing it.")
    validate_parser.add_argument("input", type=Path)
    validate_parser.add_argument(
        "--config",
        type=Path,
        help="Also verify that the HDF5 features satisfy a robot configuration.",
    )

    convert_parser = commands.add_parser("convert", help="Write a canonical RelAIBotiX HDF5 copy.")
    convert_parser.add_argument("input", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--overwrite", action="store_true")
    return parser


def _run_h5(arguments: Sequence[str]) -> int:
    args = _h5_parser().parse_args(arguments)
    if args.h5_command == "inspect":
        _print_summary(args.input)
        return 0
    if args.h5_command == "validate":
        return 0 if _print_validation(args.input, args.config) else 1

    output = convert_h5(args.input, args.output, overwrite=args.overwrite)
    print(f"Converted: {args.input} -> {output}")
    return 0 if _print_validation(output) else 1


def _config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix config",
        description="Validate a RelAIBotiX robot configuration.",
    )
    commands = parser.add_subparsers(dest="config_command", required=True)
    validate_parser = commands.add_parser("validate", help="Validate a robot JSON file.")
    validate_parser.add_argument("input", type=Path)
    return parser


def _run_config(arguments: Sequence[str]) -> int:
    from .reliability import load_robot_config

    args = _config_parser().parse_args(arguments)
    config = load_robot_config(args.input)
    print(f"VALID: {args.input}")
    print(f"  Robot: {config.name} ({config.robot_id})")
    print(f"  Components: {len(config.components)}")
    print(f"  Measured components: {config.measured_component_count}")
    print(f"  Failure probability source: {config.probability_source}")
    print(f"  Exposure assumption source: {config.exposure_assumptions.source}")
    return 0


def _behavior_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix behavior",
        description="Calculate behavioral metrics from a labeled canonical HDF5 file.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Robot configuration containing expert-provided exposure assumptions.",
    )
    parser.add_argument(
        "--skill-labels",
        default=None,
        help=(
            "Override the skill-label dataset. By default, filtered predictions, "
            "raw predictions, then ground truth are tried in that order."
        ),
    )
    return parser


def _run_behavior(arguments: Sequence[str]) -> int:
    args = _behavior_parser().parse_args(arguments)
    analyzer = BehavioralAnalyzer()
    if args.config is not None:
        from .reliability import load_robot_config

        robot_config = load_robot_config(args.config)
        assumptions = robot_config.exposure_assumptions
        from .behavioral import BehavioralThresholds

        analyzer = BehavioralAnalyzer(
            thresholds=BehavioralThresholds(
                position_step=assumptions.position_step,
                velocity_active=assumptions.velocity_active,
                effort_active=assumptions.effort_active,
                velocity_bands=assumptions.velocity_bands,
                effort_bands=assumptions.effort_bands,
            ),
            component_features={
                name: component.features
                for name, component in robot_config.components.items()
                if any(component.features.values())
            },
        )
    result = analyzer.analyze_h5(
        args.input,
        skill_labels_dataset=args.skill_labels,
    )
    result.write_csv(args.output)
    result.write_json(args.output / "behavior.json")
    print(f"Behavioral analysis: {len(result.segments)} segments")
    print(f"Results: {args.output}")
    return 0


def _skills_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix skills infer",
        description="Run a pretrained skill detector on a canonical HDF5 file.",
    )
    parser.add_argument("input", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--checkpoint", type=Path, help="Use an explicit checkpoint path.")
    selection.add_argument("--detector", help="Use a detector ID from the checkpoint registry.")
    parser.add_argument("--case-study", help="Limit automatic selection to one case study.")
    parser.add_argument("--registry", type=Path, help="Use a custom checkpoint registry JSON.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Directory containing the registered checkpoint subdirectories.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modality",
        choices=("auto", "timeseries", "camera", "hybrid"),
        default="auto",
    )
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--minimum-skill-frames", type=int, default=5)
    parser.add_argument("--target-stride", type=int, default=1)
    return parser


def _skills_list_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix skills list",
        description="List registered pretrained skill detectors.",
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    return parser


def _run_skills(arguments: Sequence[str]) -> int:
    from .skilldetector import load_registry, resolve_checkpoint, select_detector

    if arguments and arguments[0] == "list":
        args = _skills_list_parser().parse_args(arguments[1:])
        registry = load_registry(args.registry)
        for detector in registry.detectors.values():
            try:
                checkpoint = resolve_checkpoint(detector, args.checkpoint_root)
                availability = str(checkpoint)
            except FileNotFoundError:
                availability = "checkpoint not installed"
            default = " [recommended]" if detector.recommended else ""
            print(
                f"{detector.detector_id}: {detector.case_study}, "
                f"{detector.modality}{default} - {availability}"
            )
        return 0
    if not arguments or arguments[0] != "infer":
        raise SystemExit("Usage: relaibotix skills {list,infer} ...")
    args = _skills_parser().parse_args(arguments[1:])
    from .skilldetector import run_inference

    detector = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        modality = "timeseries" if args.modality == "auto" else args.modality
    else:
        registry = load_registry(args.registry)
        detector = select_detector(
            registry,
            args.input,
            detector_id=args.detector,
            case_study=args.case_study,
            modality=None if args.modality == "auto" else args.modality,
        )
        checkpoint = resolve_checkpoint(detector, args.checkpoint_root)
        modality = detector.modality
        print(f"Selected detector: {detector.detector_id}")

    result = run_inference(
        h5_path=args.input,
        checkpoint_path=checkpoint,
        output_h5=args.output,
        modality=modality,
        lerobot_root=args.lerobot_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        minimum_skill_frames=args.minimum_skill_frames,
        target_stride=args.target_stride,
    )
    print(
        f"Skill inference: {result.samples} samples across {result.episodes} episodes "
        f"-> {result.output_h5}"
    )
    return 0


def _reliability_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix reliability",
        description="Build per-skill fault trees from behavioral exposure.",
    )
    parser.add_argument("behavior", type=Path, help="behavior.json produced by the behavior command")
    parser.add_argument("--config", type=Path, required=True, help="Robot reliability JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prism", action="store_true", help="Verify models with PRISM")
    parser.add_argument("--prism-executable", default="prism")
    parser.add_argument("--storm", action="store_true", help="Verify the PRISM model with STORM")
    parser.add_argument("--storm-executable", default="storm")
    parser.add_argument(
        "--approximate-solvers",
        action="store_true",
        help="Use approximate external model checking instead of publication-grade exact mode.",
    )
    parser.add_argument(
        "--sensitivity",
        nargs="?",
        type=float,
        const=10.0,
        metavar="FACTOR",
        help="Perturb each component probability one at a time (default factor: 10).",
    )
    return parser


def _run_reliability(arguments: Sequence[str]) -> int:
    from .reliability import analyze_reliability, load_robot_config
    from .reliability.prism import write_prism_and_props, write_prism_no_done_and_props

    args = _reliability_parser().parse_args(arguments)
    behavior = BehavioralResult.read_json(args.behavior)
    config = load_robot_config(args.config)
    result = analyze_reliability(behavior, config)
    args.output.mkdir(parents=True, exist_ok=True)
    result.component_failures.to_csv(args.output / "component_failures.csv", index=False)
    result.skill_probabilities.to_csv(args.output / "skill_probabilities.csv", index=False)
    result.write_json(args.output / "reliability.json")
    if args.sensitivity is not None:
        from .reliability import analyze_component_sensitivity

        sensitivity = analyze_component_sensitivity(
            behavior,
            config,
            factor=args.sensitivity,
            baseline=result,
        )
        sensitivity.to_csv(args.output / "sensitivity.csv", index=False)
        (args.output / "sensitivity.json").write_text(
            sensitivity.to_json(orient="records", indent=2) + "\n"
        )
        if not sensitivity.empty:
            print(f"Most influential component: {sensitivity.iloc[0]['component']}")
    prism_path, properties_path = write_prism_and_props(result.dtmc, args.output / "model")
    repeated_path, repeated_properties = write_prism_no_done_and_props(
        result.dtmc,
        args.output / "model_repeated_runs",
        state_time_seconds=dict(result.repeated_run_mttf.state_time_seconds),
        start_state="start",
        precision=20,
    )
    exact = not args.approximate_solvers

    def verify(values, expected, solver_name):
        # Long-horizon rewards make the double-precision internal linear solve
        # mildly ill-conditioned. Exact external results still agree to at least
        # eight significant digits for the publication models.
        if len(values) < len(expected) or any(
            not math.isclose(actual, reference, rel_tol=2e-8, abs_tol=1e-12)
            for actual, reference in zip(values, expected)
        ):
            raise RuntimeError(f"{solver_name} and the internal solver disagree.")

    expected = (
        result.dtmc_solution.failure_probability,
        result.dtmc_solution.completion_without_modeled_failure_probability,
    )
    repeated_expected = (1.0, result.repeated_run_mttf.seconds)
    if args.prism:
        from .reliability import run_prism

        prism_result = run_prism(
            prism_path,
            properties_path,
            executable=args.prism_executable,
            exact=exact,
        )
        verify(prism_result.values[:2], expected, "PRISM")
        prism_result.write_json(args.output / "prism.json")
        prism_mttf = run_prism(
            repeated_path,
            repeated_properties,
            executable=args.prism_executable,
            exact=exact,
        )
        verify(prism_mttf.values, repeated_expected, "PRISM repeated-run MTTF")
        prism_mttf.write_json(args.output / "prism_mttf.json")
        print(f"PRISM verified {len(prism_result.values) + len(prism_mttf.values)} properties")
    if args.storm:
        from .reliability import run_storm

        storm_result = run_storm(
            prism_path,
            properties_path,
            executable=args.storm_executable,
            exact=exact,
        )
        verify(storm_result.values[:2], expected, "STORM")
        storm_result.write_json(args.output / "storm.json")
        storm_mttf = run_storm(
            repeated_path,
            repeated_properties,
            executable=args.storm_executable,
            exact=exact,
        )
        verify(storm_mttf.values, repeated_expected, "STORM repeated-run MTTF")
        storm_mttf.write_json(args.output / "storm_mttf.json")
        print(f"STORM verified {len(storm_result.values) + len(storm_mttf.values)} properties")
    print(f"Reliability analysis: {len(result.skill_probabilities)} skills")
    print(f"System failure probability: {result.dtmc_solution.failure_probability:.12g}")
    print(
        "Completion without modeled failure: "
        f"{result.dtmc_solution.completion_without_modeled_failure_probability:.12g}"
    )
    print(f"Repeated-run MTTF: {result.repeated_run_mttf.hours:.12g} h")
    print(f"Results: {args.output}")
    return 0


def _experiments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix experiments",
        description="Reproduce a declared set of publication experiments.",
    )
    commands = parser.add_subparsers(dest="experiments_command", required=True)
    run_parser = commands.add_parser("run", help="Run every experiment in a manifest.")
    run_parser.add_argument("manifest", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--prism", action="store_true")
    run_parser.add_argument("--prism-executable", default="prism")
    run_parser.add_argument("--storm", action="store_true")
    run_parser.add_argument("--storm-executable", default="storm")
    run_parser.add_argument("--approximate-solvers", action="store_true")
    return parser


def _run_experiments(arguments: Sequence[str]) -> int:
    from .experiments import (
        configured_analyzer,
        load_experiment_manifest,
        write_experiment_summary,
    )
    from .reliability import load_robot_config

    args = _experiments_parser().parse_args(arguments)
    manifest = load_experiment_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "Publication regression: using detector predictions already stored in the "
        "declared legacy HDF5 files."
    )
    for index, experiment in enumerate(manifest.experiments, start=1):
        print(
            f"[{index}/{len(manifest.experiments)}] "
            f"{experiment.setting} / {experiment.task} / {experiment.policy}"
        )
        if not _print_validation(experiment.input_h5, experiment.robot_config):
            raise ValueError(f"Invalid experiment input: {experiment.experiment_id}")
        config = load_robot_config(experiment.robot_config)
        result = configured_analyzer(config, experiment.skill_names).analyze_h5(
            experiment.input_h5
        )
        behavior_output = args.output / experiment.experiment_id / "behavior"
        result.write_csv(behavior_output)
        result.write_json(behavior_output / "behavior.json")

        reliability_arguments = [
            str(behavior_output / "behavior.json"),
            "--config",
            str(experiment.robot_config),
            "--output",
            str(args.output / experiment.experiment_id / "reliability"),
            "--sensitivity",
        ]
        if args.prism:
            reliability_arguments.extend(
                ("--prism", "--prism-executable", args.prism_executable)
            )
        if args.storm:
            reliability_arguments.extend(
                ("--storm", "--storm-executable", args.storm_executable)
            )
        if args.approximate_solvers:
            reliability_arguments.append("--approximate-solvers")
        _run_reliability(reliability_arguments)

    solver_metadata: dict[str, object] = {
        "internal": {
            "enabled": True,
            "method": "linear_system_double_precision",
        }
    }
    first_output = args.output / manifest.experiments[0].experiment_id / "reliability"
    for solver, enabled in (("prism", args.prism), ("storm", args.storm)):
        metadata: dict[str, object] = {"enabled": enabled}
        if enabled:
            result = json.loads((first_output / f"{solver}.json").read_text())
            metadata.update({
                "version": result["version"],
                "exact": result["exact"],
                "verified_experiments": len(manifest.experiments),
            })
        solver_metadata[solver] = metadata
    paths = write_experiment_summary(
        manifest,
        args.output,
        solver_metadata=solver_metadata,
    )
    print(f"Experiment set complete: {len(manifest.experiments)} experiments")
    print(f"Publication table: {paths[0]}")
    return 0


def _pipeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix run",
        description="Run the complete RelAIBotiX analysis pipeline from one HDF5 input.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--checkpoint", type=Path)
    selection.add_argument("--detector")
    parser.add_argument("--case-study")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--modality",
        choices=("auto", "timeseries", "camera", "hybrid"),
        default="auto",
    )
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--minimum-skill-frames", type=int, default=5)
    parser.add_argument("--target-stride", type=int, default=1)
    parser.add_argument(
        "--sensitivity",
        nargs="?",
        type=float,
        const=10.0,
        metavar="FACTOR",
    )
    parser.add_argument("--prism", action="store_true")
    parser.add_argument("--prism-executable", default="prism")
    parser.add_argument("--storm", action="store_true")
    parser.add_argument("--storm-executable", default="storm")
    parser.add_argument("--approximate-solvers", action="store_true")
    return parser


def _run_pipeline(arguments: Sequence[str]) -> int:
    args = _pipeline_parser().parse_args(arguments)
    if not _print_validation(args.input, args.config):
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    source = args.input
    if inspect_h5(source).layout == "flat":
        source = args.output / "canonical.h5"
        if source.exists():
            raise FileExistsError(f"Pipeline output already exists: {source}")
        convert_h5(args.input, source)
        print(f"Canonical input: {source}")

    predicted = args.output / "predicted.h5"
    if predicted.exists():
        raise FileExistsError(f"Pipeline output already exists: {predicted}")
    skill_arguments = [
        "infer",
        str(source),
        "--output",
        str(predicted),
        "--modality",
        args.modality,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--minimum-skill-frames",
        str(args.minimum_skill_frames),
        "--target-stride",
        str(args.target_stride),
    ]
    for option, value in (
        ("--checkpoint", args.checkpoint),
        ("--detector", args.detector),
        ("--case-study", args.case_study),
        ("--registry", args.registry),
        ("--checkpoint-root", args.checkpoint_root),
        ("--lerobot-root", args.lerobot_root),
    ):
        if value is not None:
            skill_arguments.extend((option, str(value)))
    _run_skills(skill_arguments)

    if not _print_validation(predicted, args.config):
        raise RuntimeError("Detector output failed HDF5 validation.")

    behavior_directory = args.output / "behavior"
    _run_behavior([
        str(predicted),
        "--config",
        str(args.config),
        "--output",
        str(behavior_directory),
    ])

    reliability_arguments = [
        str(behavior_directory / "behavior.json"),
        "--config",
        str(args.config),
        "--output",
        str(args.output / "reliability"),
    ]
    if args.sensitivity is not None:
        reliability_arguments.extend(("--sensitivity", str(args.sensitivity)))
    if args.prism:
        reliability_arguments.extend(("--prism", "--prism-executable", args.prism_executable))
    if args.storm:
        reliability_arguments.extend(("--storm", "--storm-executable", args.storm_executable))
    if args.approximate_solvers:
        reliability_arguments.append("--approximate-solvers")
    _run_reliability(reliability_arguments)
    print(f"Pipeline complete: {args.output}")
    return 0


def _top_level_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix",
        description="Validate data, run skill inference, and analyze robot behavior.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("h5", "config", "skills", "behavior", "reliability", "experiments", "run"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        _top_level_parser().print_help()
        return 0
    if arguments and arguments[0] == "h5":
        return _run_h5(arguments[1:])
    if arguments and arguments[0] == "config":
        return _run_config(arguments[1:])
    if arguments and arguments[0] == "behavior":
        return _run_behavior(arguments[1:])
    if arguments and arguments[0] == "skills":
        return _run_skills(arguments[1:])
    if arguments and arguments[0] == "reliability":
        return _run_reliability(arguments[1:])
    if arguments and arguments[0] == "experiments":
        return _run_experiments(arguments[1:])
    if arguments and arguments[0] == "run":
        return _run_pipeline(arguments[1:])
    _top_level_parser().parse_args(arguments)
    return 0
