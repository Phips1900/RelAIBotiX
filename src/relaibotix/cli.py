"""Command-line interface for RelAIBotiX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

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


def _run_legacy_analysis(arguments: Sequence[str]) -> int:
    """Keep the existing pipeline callable until the new pipeline replaces it."""

    from .handler import _cli_relaibotix

    previous_argv = sys.argv
    try:
        sys.argv = ["relaibotix", *arguments]
        _cli_relaibotix()
    finally:
        sys.argv = previous_argv
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "h5":
        return _run_h5(arguments[1:])
    if arguments and arguments[0] == "analyze":
        return _run_legacy_analysis(arguments[1:])
    return _run_legacy_analysis(arguments)
