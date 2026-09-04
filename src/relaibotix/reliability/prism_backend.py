"""PRISM command-line backend for generated reliability models."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess


@dataclass(frozen=True)
class PrismResult:
    executable: str
    version: str | None
    exact: bool
    properties: tuple[str, ...]
    values: tuple[float, ...]
    stdout: str

    def write_json(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.write_text(json.dumps({
            "executable": self.executable,
            "version": self.version,
            "exact": self.exact,
            "results": [
                {"property": prop, "value": value}
                for prop, value in zip(self.properties, self.values)
            ],
        }, indent=2) + "\n")
        return destination


def _properties(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    )


def run_prism(
    model_path: str | Path,
    properties_path: str | Path,
    *,
    executable: str = "prism",
    exact: bool = True,
    timeout: float = 120.0,
) -> PrismResult:
    """Run PRISM and parse one result for every property."""

    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(
            f"PRISM executable '{executable}' was not found. Install PRISM or provide "
            "--prism-executable."
        )
    model = Path(model_path)
    properties_file = Path(properties_path)
    properties = _properties(properties_file)
    command = [resolved, str(model), str(properties_file)]
    if exact:
        command.append("-exact")
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"PRISM failed with exit code {completed.returncode}: {detail}"
        )
    raw_values = re.findall(r"^Result:\s*([^\s]+)", completed.stdout, re.MULTILINE)
    if len(raw_values) != len(properties):
        raise RuntimeError(
            f"PRISM returned {len(raw_values)} results for {len(properties)} properties."
        )
    try:
        values = tuple(float(Fraction(value)) for value in raw_values)
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"PRISM returned a non-numeric result: {raw_values}") from error
    version_match = re.search(r"^Version:\s*([^\r\n]+)", completed.stdout, re.MULTILINE)
    return PrismResult(
        executable=resolved,
        version=version_match.group(1).strip() if version_match else None,
        exact=exact,
        properties=properties,
        values=values,
        stdout=completed.stdout,
    )
