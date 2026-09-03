"""Robot reliability configuration loading and fault-tree construction."""

from __future__ import annotations

from dataclasses import dataclass, field
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
        if not self.source.strip():
            raise ValueError("Exposure assumption source must not be empty.")
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
    component_type: str
    failure_probability: float
    redundancy_copies: int = 1
    redundancy_mode: str = "parallel"
    features: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    always_active: bool = False
    behavior_sources: tuple[str, ...] = ()
    exposure: str = "skill_time"
    distance_thresholds: tuple[float, float] | None = None
    distance_unit: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.component_type:
            raise ValueError("Component name and type must not be empty.")
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError(f"Failure probability for '{self.name}' must lie in [0, 1].")
        if self.redundancy_copies < 1:
            raise ValueError(f"Redundancy copies for '{self.name}' must be at least one.")
        if self.redundancy_mode != "parallel":
            raise ValueError(
                f"Redundancy mode for '{self.name}' must currently be 'parallel'."
            )
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
    robot_id: str
    name: str
    robot_type: str
    components: Mapping[str, ComponentConfig]
    probability_basis: str = "per_minute"
    probability_source: str = "requires_expert_review"
    exposure_assumptions: ExposureAssumptions = ExposureAssumptions()

    @property
    def measured_component_count(self) -> int:
        """Number of reliability components backed by recorded telemetry."""

        return sum(any(component.features.values()) for component in self.components.values())

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
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Component '{component_name}' requires redundancy as an object with copies and mode."
        )
    if "copies" not in value:
        raise ValueError(f"Component '{component_name}' redundancy requires 'copies'.")
    raw_copies = value["copies"]
    if isinstance(raw_copies, bool) or not isinstance(raw_copies, int):
        raise ValueError(f"Component '{component_name}' redundancy copies must be an integer.")
    copies = raw_copies
    mode = str(value.get("mode", "parallel"))
    if mode != "parallel":
        raise ValueError(
            f"Component '{component_name}' redundancy mode must currently be 'parallel'."
        )
    return copies


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
    if not isinstance(raw, Mapping):
        raise ValueError("exposure_assumptions must be an object.")
    required = {
        "source",
        "position_step",
        "velocity_active",
        "effort_active",
        "velocity_bands",
        "effort_bands",
        "velocity_multipliers",
        "effort_multipliers",
        "distance_multipliers",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            "exposure_assumptions is missing required fields: " + ", ".join(missing)
        )
    return ExposureAssumptions(
        position_step=float(raw["position_step"]),
        velocity_active=float(raw["velocity_active"]),
        effort_active=float(raw["effort_active"]),
        velocity_bands=_pair(raw, "velocity_bands", (0.0, 0.0)),
        effort_bands=_pair(raw, "effort_bands", (0.0, 0.0)),
        velocity_multipliers=_triple(raw, "velocity_multipliers", (0.0, 0.0, 0.0)),
        effort_multipliers=_triple(raw, "effort_multipliers", (0.0, 0.0, 0.0)),
        distance_multipliers=_triple(raw, "distance_multipliers", (0.0, 0.0, 0.0)),
        source=str(raw["source"]),
    )


def load_robot_config(path: str | Path) -> RobotConfig:
    """Load and validate the supported robot configuration schema."""

    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    if raw.get("schema_version") != "1.0":
        raise ValueError("Robot configuration schema_version must be '1.0'.")
    robot = raw.get("robot")
    if not isinstance(robot, Mapping):
        raise ValueError("Robot configuration requires a 'robot' object.")
    missing_robot_fields = sorted({"id", "name", "type"} - set(robot))
    if missing_robot_fields:
        raise ValueError(
            "Robot identity is missing required fields: " + ", ".join(missing_robot_fields)
        )
    for field_name in ("id", "name", "type"):
        if not isinstance(robot[field_name], str) or not robot[field_name].strip():
            raise ValueError(f"Robot identity field '{field_name}' must be a non-empty string.")
    probability_basis = raw.get("failure_probability_basis")
    if probability_basis not in {"per_second", "per_minute", "per_hour"}:
        raise ValueError(
            "failure_probability_basis must be per_second, per_minute, or per_hour."
        )
    probability_source = raw.get("failure_probability_source")
    if not isinstance(probability_source, str) or not probability_source.strip():
        raise ValueError("Robot configuration requires failure_probability_source.")
    raw_components = raw.get("components")
    if not isinstance(raw_components, dict) or not raw_components:
        raise ValueError("Robot configuration requires a non-empty 'components' object.")
    components: dict[str, ComponentConfig] = {}
    for name, definition in raw_components.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Component '{name}' must be an object.")
        missing_component_fields = sorted(
            {"type", "failure_probability", "redundancy"} - set(definition)
        )
        if missing_component_fields:
            raise ValueError(
                f"Component '{name}' is missing required fields: "
                + ", ".join(missing_component_fields)
            )
        raw_features = definition.get("features", {})
        if not isinstance(raw_features, Mapping):
            raise ValueError(f"Component '{name}' features must be an object.")
        features: dict[str, tuple[str, ...]] = {}
        for signal, feature_names in raw_features.items():
            if feature_names is None:
                normalized_features: tuple[str, ...] = ()
            elif isinstance(feature_names, str):
                normalized_features = (feature_names,)
            elif isinstance(feature_names, list) and all(
                isinstance(feature_name, str) for feature_name in feature_names
            ):
                normalized_features = tuple(feature_names)
            else:
                raise ValueError(
                    f"Component '{name}' feature '{signal}' must be a string, list of strings, or null."
                )
            features[str(signal)] = normalized_features
        normalized = str(name).lower()
        joint_match = re.fullmatch(r"joint[_ -]?(\d+)", normalized)
        inferred_sources = (f"j{joint_match.group(1)}",) if joint_match else ()
        if normalized == "gripper":
            inferred_sources = ("gripper",)
        if not inferred_sources and any(features.values()):
            inferred_sources = (normalized,)
        raw_sources = definition.get("behavior_sources", inferred_sources)
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        if not isinstance(raw_sources, (list, tuple)):
            raise ValueError(f"Component '{name}' behavior_sources must be a list.")
        raw_always_active = definition.get("always_active", False)
        if not isinstance(raw_always_active, bool):
            raise ValueError(f"Component '{name}' always_active must be Boolean.")
        always_active = raw_always_active
        if not always_active and not any(features.values()):
            raise ValueError(
                f"Component '{name}' must define measured features or set always_active to true."
            )
        exposure = "skill_time" if always_active else "motion"
        raw_distance_thresholds = definition.get("distance_thresholds")
        distance_thresholds = None
        if raw_distance_thresholds is not None:
            if not isinstance(raw_distance_thresholds, (list, tuple)) or len(raw_distance_thresholds) != 2:
                raise ValueError(
                    f"Component '{name}' distance_thresholds must contain medium and high values."
                )
            distance_thresholds = tuple(map(float, raw_distance_thresholds))
            if not definition.get("distance_unit"):
                raise ValueError(
                    f"Component '{name}' requires distance_unit with distance_thresholds."
                )
        components[str(name)] = ComponentConfig(
            name=str(name),
            component_type=str(definition["type"]),
            failure_probability=float(definition["failure_probability"]),
            redundancy_copies=_redundancy_copies(definition.get("redundancy"), str(name)),
            redundancy_mode=str(definition["redundancy"].get("mode", "parallel")),
            features=features,
            always_active=always_active,
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
        robot_id=str(robot["id"]),
        name=str(robot["name"]),
        robot_type=str(robot["type"]),
        components=components,
        probability_basis=str(probability_basis),
        probability_source=probability_source,
        exposure_assumptions=_exposure_assumptions(raw.get("exposure_assumptions")),
    )
