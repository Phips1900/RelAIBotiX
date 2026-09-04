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
    version: str | None = None
    exact: bool = False

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
    command = [resolved, "--prism", str(model), "--prop", "; ".join(properties)]
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
        raise RuntimeError(
            f"STORM failed with exit code {completed.returncode}: {detail}"
        )
    raw_values = re.findall(
        r"Result(?:\s+\((?:for )?initial states\))?:\s*([^\r\n]+)",
        completed.stdout,
    )
    if len(raw_values) != len(properties):
        raise RuntimeError(
            f"STORM returned {len(raw_values)} results for {len(properties)} properties."
        )
    try:
        values = tuple(float(Fraction(value.strip().split()[0])) for value in raw_values)
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"STORM returned a non-numeric result: {raw_values}") from error
    stdout = completed.stdout
    version_match = re.search(r"^Storm\s+([^\s]+)", stdout, re.MULTILINE)
    return StormResult(
        resolved,
        properties,
        values,
        stdout,
        version_match.group(1) if version_match else None,
        exact,
    )
