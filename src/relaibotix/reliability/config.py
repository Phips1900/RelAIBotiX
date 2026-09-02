"""Robot reliability configuration loading and fault-tree construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

from .fault_tree import FaultTreeModel, Gate


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    failure_probability: float
    redundancy_copies: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError(f"Failure probability for '{self.name}' must lie in [0, 1].")
        if self.redundancy_copies < 1:
            raise ValueError(f"Redundancy copies for '{self.name}' must be at least one.")


@dataclass(frozen=True)
class RobotConfig:
    name: str
    robot_type: str
    components: Mapping[str, ComponentConfig]
    probability_basis: str = "per_minute"

    @property
    def joint_count(self) -> int:
        return sum(bool(re.fullmatch(r"joint[_ -]?\d+", name, re.IGNORECASE)) for name in self.components)

    @property
    def redundant_components(self) -> Mapping[str, int]:
        return {
            name: component.redundancy_copies
            for name, component in self.components.items()
            if component.redundancy_copies > 1
        }

    def build_fault_tree(
        self,
        *,
        component_probabilities: Mapping[str, float] | None = None,
        active_components: list[str] | tuple[str, ...] | None = None,
        top_event: str = "system_failure",
    ) -> FaultTreeModel:
        """Build the standard OR-of-components tree with explicit redundant copies."""

        selected = tuple(active_components) if active_components is not None else tuple(self.components)
        unknown = set(selected) - set(self.components)
        if unknown:
            raise ValueError(f"Unknown active components: {sorted(unknown)}")
        supplied = dict(component_probabilities or {})
        unknown_probabilities = set(supplied) - set(self.components)
        if unknown_probabilities:
            raise ValueError(f"Probabilities supplied for unknown components: {sorted(unknown_probabilities)}")

        basic_events: dict[str, float] = {}
        gates: dict[str, Gate] = {}
        top_children: list[str] = []
        for name in selected:
            component = self.components[name]
            probability = float(supplied.get(name, component.failure_probability))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Failure probability for '{name}' must lie in [0, 1].")
            if component.redundancy_copies == 1:
                basic_events[name] = probability
                top_children.append(name)
                continue
            copies = tuple(f"{name}_{index}" for index in range(1, component.redundancy_copies + 1))
            basic_events.update({copy: probability for copy in copies})
            gate_name = f"loss_of_{name}"
            gates[gate_name] = Gate("AND", copies)
            top_children.append(gate_name)

        if not top_children:
            raise ValueError("A fault tree requires at least one active component.")
        gates[top_event] = Gate("OR", tuple(top_children))
        return FaultTreeModel(top_event, basic_events, gates)


def _redundancy_copies(value: object, component_name: str) -> int:
    if isinstance(value, bool):
        return 2 if value else 1
    if isinstance(value, Mapping):
        enabled = bool(value.get("enabled", True))
        copies = int(value.get("copies", 2 if enabled else 1))
        return copies if enabled else 1
    if value is None:
        return 1
    raise ValueError(f"Invalid redundancy definition for '{component_name}'.")


def load_robot_config(path: str | Path) -> RobotConfig:
    """Load current robot JSON files, including the legacy Boolean redundancy form."""

    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    raw_components = raw.get("components")
    if not isinstance(raw_components, dict) or not raw_components:
        raise ValueError("Robot configuration requires a non-empty 'components' object.")
    components: dict[str, ComponentConfig] = {}
    for name, definition in raw_components.items():
        if not isinstance(definition, dict) or "failure_probability" not in definition:
            raise ValueError(f"Component '{name}' requires a failure_probability.")
        components[str(name)] = ComponentConfig(
            name=str(name),
            failure_probability=float(definition["failure_probability"]),
            redundancy_copies=_redundancy_copies(definition.get("redundancy"), str(name)),
        )
    return RobotConfig(
        name=str(raw.get("robot", config_path.stem)),
        robot_type=str(raw.get("robot_type", "unknown")),
        components=components,
        probability_basis=str(raw.get("failure_probability_basis", "per_minute")),
    )
