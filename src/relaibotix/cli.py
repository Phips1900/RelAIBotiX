"""Command-line interface for RelAIBotiX."""

from __future__ import annotations

import argparse
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


def _print_validation(path: str | Path) -> bool:
    report = validate_h5(path)
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
        return 0 if _print_validation(args.input) else 1

    output = convert_h5(args.input, args.output, overwrite=args.overwrite)
    print(f"Converted: {args.input} -> {output}")
    return 0 if _print_validation(output) else 1


def _behavior_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix behavior",
        description="Calculate behavioral metrics from a labeled canonical HDF5 file.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
    result = BehavioralAnalyzer().analyze_h5(
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modality", choices=("timeseries", "camera", "hybrid"), default="timeseries")
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--minimum-skill-frames", type=int, default=5)
    parser.add_argument("--target-stride", type=int, default=1)
    return parser


def _run_skills(arguments: Sequence[str]) -> int:
    if not arguments or arguments[0] != "infer":
        raise SystemExit("Usage: relaibotix skills infer ...")
    args = _skills_parser().parse_args(arguments[1:])
    from .skilldetector import run_inference

    result = run_inference(
        h5_path=args.input,
        checkpoint_path=args.checkpoint,
        output_h5=args.output,
        modality=args.modality,
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
    parser.add_argument("--storm", action="store_true", help="Verify the PRISM model with STORM")
    parser.add_argument("--storm-executable", default="storm")
    parser.add_argument("--storm-exact", action="store_true")
    return parser


def _run_reliability(arguments: Sequence[str]) -> int:
    from .reliability import analyze_reliability, load_robot_config
    from .reliability.prism import write_prism_and_props

    args = _reliability_parser().parse_args(arguments)
    result = analyze_reliability(
        BehavioralResult.read_json(args.behavior),
        load_robot_config(args.config),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result.component_failures.to_csv(args.output / "component_failures.csv", index=False)
    result.skill_probabilities.to_csv(args.output / "skill_probabilities.csv", index=False)
    result.write_json(args.output / "reliability.json")
    prism_path, properties_path = write_prism_and_props(result.dtmc, args.output / "model")
    if args.storm:
        from .reliability import run_storm

        storm_result = run_storm(
            prism_path,
            properties_path,
            executable=args.storm_executable,
            exact=args.storm_exact,
        )
        expected = (
            result.dtmc_solution.failure_probability,
            result.dtmc_solution.success_probability,
        )
        if len(storm_result.values) >= 2 and any(
            not math.isclose(actual, reference, rel_tol=1e-8, abs_tol=1e-12)
            for actual, reference in zip(storm_result.values[:2], expected)
        ):
            raise RuntimeError(
                "STORM and the internal DTMC solver disagree on failure or success probability."
            )
        storm_result.write_json(args.output / "storm.json")
        print(f"STORM verified {len(storm_result.values)} properties")
    print(f"Reliability analysis: {len(result.skill_probabilities)} skills")
    print(f"System failure probability: {result.dtmc_solution.failure_probability:.12g}")
    print(f"System success probability: {result.dtmc_solution.success_probability:.12g}")
    print(f"Results: {args.output}")
    return 0


def _top_level_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relaibotix",
        description="Validate data, run skill inference, and analyze robot behavior.",
    )
    parser.add_argument("command", nargs="?", choices=("h5", "skills", "behavior", "reliability"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        _top_level_parser().print_help()
        return 0
    if arguments and arguments[0] == "h5":
        return _run_h5(arguments[1:])
    if arguments and arguments[0] == "behavior":
        return _run_behavior(arguments[1:])
    if arguments and arguments[0] == "skills":
        return _run_skills(arguments[1:])
    if arguments and arguments[0] == "reliability":
        return _run_reliability(arguments[1:])
    _top_level_parser().parse_args(arguments)
    return 0
