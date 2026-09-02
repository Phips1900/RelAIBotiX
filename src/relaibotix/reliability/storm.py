"""Optional STORM command-line backend for generated PRISM models."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess


@dataclass(frozen=True)
class StormResult:
    executable: str
    properties: tuple[str, ...]
    values: tuple[float, ...]
    stdout: str

    def write_json(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.write_text(json.dumps({
            "executable": self.executable,
            "results": [
                {"property": prop, "value": value}
                for prop, value in zip(self.properties, self.values)
            ],
        }, indent=2) + "\n")
        return destination


def run_storm(
    prism_path: str | Path,
    properties_path: str | Path,
    *,
    executable: str = "storm",
    exact: bool = False,
    timeout: float = 120.0,
) -> StormResult:
    """Run STORM on a PRISM model and return numeric property results."""

    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(
            f"STORM executable '{executable}' was not found. Install STORM or provide "
            "--storm-executable."
        )
    model = Path(prism_path)
    properties_file = Path(properties_path)
    properties = tuple(
        line.strip()
        for line in properties_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    )
    command = [resolved, "--prism", str(model), "--prop", str(properties_file)]
    if exact:
        command.append("--exact")
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"STORM failed with exit code {completed.returncode}: {detail}")
    raw_values = re.findall(
        r"Result(?:\s+\(initial states\))?:\s*([^\r\n]+)",
        completed.stdout,
    )
    try:
        values = tuple(float(Fraction(value.strip().split()[0])) for value in raw_values)
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"STORM returned a non-numeric result: {raw_values}") from error
    if len(values) != len(properties):
        raise RuntimeError(
            f"STORM returned {len(values)} numeric results for {len(properties)} properties."
        )
    return StormResult(resolved, properties, values, completed.stdout)
