"""Structured outputs from behavioral analysis."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BehavioralResult:
    """Tidy behavioral tables at segment and aggregated levels."""

    segments: pd.DataFrame
    joint_metrics: pd.DataFrame
    skill_summary: pd.DataFrame
    joint_summary: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "segments": self.segments,
            "joint_metrics": self.joint_metrics,
            "skill_summary": self.skill_summary,
            "joint_summary": self.joint_summary,
        }

    def write_csv(self, output_directory: str | Path) -> Path:
        """Write each result table to one CSV file."""

        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        for name, table in self.tables().items():
            table.to_csv(output / f"{name}.csv", index=False)
        return output

    def write_json(self, output_path: str | Path) -> Path:
        """Write all tables to one portable JSON document."""

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: json.loads(table.to_json(orient="records"))
            for name, table in self.tables().items()
        }
        payload["metadata"] = self.metadata
        with destination.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
        return destination

    @classmethod
    def read_json(cls, input_path: str | Path) -> "BehavioralResult":
        """Load a behavioral result previously written by :meth:`write_json`."""

        payload = json.loads(Path(input_path).read_text())
        required = ("segments", "joint_metrics", "skill_summary", "joint_summary")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"Behavior JSON is missing tables: {', '.join(missing)}")
        return cls(
            **{name: pd.DataFrame(payload[name]) for name in required},
            metadata=dict(payload.get("metadata", {})),
        )
