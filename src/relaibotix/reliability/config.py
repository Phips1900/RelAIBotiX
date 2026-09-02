"""Robot reliability configuration loading and fault-tree construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

from .fault_tree import FaultTreeModel, Gate


@dataclass(frozen=True)
class ExposureAssumptions:
    """Expert-provided thresholds and relative hazard multipliers."""

    position_step: float = 1e-3
    velocity_active: float = 3e-2
    effort_active: float = 1e-1
    velocity_bands: tuple[float, float] = (0.5, 1.0)
    effort_bands: tuple[float, float] = (0.2, 0.6)
    velocity_multipliers: tuple[float, float, float] = (1.0, 2.0, 5.0)
    effort_multipliers: tuple[float, float, float] = (1.0, 1.25, 1.75)
    distance_multipliers: tuple[float, float, float] = (1.0, 1.5, 2.0)
    source: str = "framework_example"

    def __post_init__(self) -> None:
        if self.position_step < 0.0 or self.velocity_active < 0.0 or self.effort_active < 0.0:
            raise ValueError("Exposure activity thresholds must be non-negative.")
        for name, thresholds in (
            ("velocity_bands", self.velocity_bands),
            ("effort_bands", self.effort_bands),
        ):
            if thresholds[0] < 0.0 or thresholds[1] <= thresholds[0]:
                raise ValueError(f"{name} must satisfy 0 <= medium < high.")
        for name, multipliers in (
            ("velocity_multipliers", self.velocity_multipliers),
            ("effort_multipliers", self.effort_multipliers),
            ("distance_multipliers", self.distance_multipliers),
        ):
            if any(value <= 0.0 for value in multipliers):
                raise ValueError(f"{name} must contain three positive values.")

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "position_step": self.position_step,
            "velocity_active": self.velocity_active,
            "effort_active": self.effort_active,
            "velocity_bands": list(self.velocity_bands),
            "effort_bands": list(self.effort_bands),
            "velocity_multipliers": list(self.velocity_multipliers),
            "effort_multipliers": list(self.effort_multipliers),
            "distance_multipliers": list(self.distance_multipliers),
        }


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    failure_probability: float
    redundancy_copies: int = 1
    behavior_sources: tuple[str, ...] = ()
    exposure: str = "skill_time"
    distance_thresholds: tuple[float, float] | None = None
    distance_unit: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError(f"Failure probability for '{self.name}' must lie in [0, 1].")
        if self.redundancy_copies < 1:
            raise ValueError(f"Redundancy copies for '{self.name}' must be at least one.")
        if self.exposure not in {"motion", "skill_time"}:
            raise ValueError(f"Exposure for '{self.name}' must be 'motion' or 'skill_time'.")
        if self.distance_thresholds is not None:
            medium, high = self.distance_thresholds
            if medium < 0.0 or high <= medium:
                raise ValueError(
                    f"Distance thresholds for '{self.name}' must satisfy 0 <= medium < high."
                )


@dataclass(frozen=True)
class RobotConfig:
    name: str
    robot_type: str
    components: Mapping[str, ComponentConfig]
    probability_basis: str = "per_minute"
    exposure_assumptions: ExposureAssumptions = ExposureAssumptions()

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


def _pair(raw: Mapping[str, object], name: str, default: tuple[float, float]) -> tuple[float, float]:
    value = raw.get(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"exposure_assumptions.{name} must contain two values.")
    return tuple(map(float, value))


def _triple(
    raw: Mapping[str, object],
    name: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    value = raw.get(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"exposure_assumptions.{name} must contain low, medium, and high values.")
    return tuple(map(float, value))


def _exposure_assumptions(raw: object) -> ExposureAssumptions:
    if raw is None:
        return ExposureAssumptions()
    if not isinstance(raw, Mapping):
        raise ValueError("exposure_assumptions must be an object.")
    defaults = ExposureAssumptions()
    return ExposureAssumptions(
        position_step=float(raw.get("position_step", defaults.position_step)),
        velocity_active=float(raw.get("velocity_active", defaults.velocity_active)),
        effort_active=float(raw.get("effort_active", defaults.effort_active)),
        velocity_bands=_pair(raw, "velocity_bands", defaults.velocity_bands),
        effort_bands=_pair(raw, "effort_bands", defaults.effort_bands),
        velocity_multipliers=_triple(
            raw, "velocity_multipliers", defaults.velocity_multipliers
        ),
        effort_multipliers=_triple(raw, "effort_multipliers", defaults.effort_multipliers),
        distance_multipliers=_triple(
            raw, "distance_multipliers", defaults.distance_multipliers
        ),
        source=str(raw.get("source", defaults.source)),
    )


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
        normalized = str(name).lower()
        joint_match = re.fullmatch(r"joint[_ -]?(\d+)", normalized)
        inferred_sources = (f"j{joint_match.group(1)}",) if joint_match else ()
        if normalized == "gripper":
            inferred_sources = ("gripper",)
        raw_sources = definition.get("behavior_sources", inferred_sources)
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        exposure = definition.get("exposure", "motion" if raw_sources else "skill_time")
        raw_distance_thresholds = definition.get("distance_thresholds")
        distance_thresholds = None
        if raw_distance_thresholds is not None:
            if not isinstance(raw_distance_thresholds, (list, tuple)) or len(raw_distance_thresholds) != 2:
                raise ValueError(
                    f"Component '{name}' distance_thresholds must contain medium and high values."
                )
            distance_thresholds = tuple(map(float, raw_distance_thresholds))
        components[str(name)] = ComponentConfig(
            name=str(name),
            failure_probability=float(definition["failure_probability"]),
            redundancy_copies=_redundancy_copies(definition.get("redundancy"), str(name)),
            behavior_sources=tuple(map(str, raw_sources)),
            exposure=str(exposure),
            distance_thresholds=distance_thresholds,
            distance_unit=(
                str(definition["distance_unit"])
                if definition.get("distance_unit") is not None
                else None
            ),
        )
    return RobotConfig(
        name=str(raw.get("robot", config_path.stem)),
        robot_type=str(raw.get("robot_type", "unknown")),
        components=components,
        probability_basis=str(raw.get("failure_probability_basis", "per_minute")),
        exposure_assumptions=_exposure_assumptions(raw.get("exposure_assumptions")),
    )
