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
    values: list[float] = []
    outputs: list[str] = []
    for prop in properties:
        command = [resolved, "--prism", str(model), "--prop", prop]
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
                f"STORM failed for property '{prop}' with exit code "
                f"{completed.returncode}: {detail}"
            )
        raw_values = re.findall(
            r"Result(?:\s+\((?:for )?initial states\))?:\s*([^\r\n]+)",
            completed.stdout,
        )
        if len(raw_values) != 1:
            raise RuntimeError(
                f"STORM returned {len(raw_values)} numeric results for property '{prop}'."
            )
        try:
            values.append(float(Fraction(raw_values[0].strip().split()[0])))
        except (ValueError, ZeroDivisionError) as error:
            raise RuntimeError(f"STORM returned a non-numeric result: {raw_values[0]}") from error
        outputs.append(completed.stdout)
    return StormResult(resolved, properties, tuple(values), "\n".join(outputs))
