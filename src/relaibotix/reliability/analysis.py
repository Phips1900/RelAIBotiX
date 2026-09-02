"""Bridge behavioral exposure to per-skill reliability models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import pandas as pd

from relaibotix.behavioral.results import BehavioralResult

from .config import RobotConfig
from .dtmc import DTMCModel, DTMCSolution, solve_dtmc
from .fault_tree import FaultTreeModel, bdd_probability, bottom_up_probability


@dataclass(frozen=True)
class ExposureMultipliers:
    velocity: tuple[float, float, float] = (1.0, 2.0, 5.0)
    effort: tuple[float, float, float] = (1.0, 1.25, 1.75)


@dataclass(frozen=True)
class ReliabilityResult:
    component_failures: pd.DataFrame
    fault_trees: dict[str, FaultTreeModel]
    skill_probabilities: pd.DataFrame
    dtmc: DTMCModel
    dtmc_solution: DTMCSolution

    def write_json(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "component_failures": json.loads(self.component_failures.to_json(orient="records")),
            "skill_probabilities": json.loads(self.skill_probabilities.to_json(orient="records")),
            "fault_trees": {
                skill: {
                    "top_event": tree.top_event,
                    "basic_events": dict(tree.basic_events),
                    "gates": {
                        name: {"operator": gate.operator, "children": list(gate.children)}
                        for name, gate in tree.gates.items()
                    },
                }
                for skill, tree in self.fault_trees.items()
            },
            "dtmc": {
                "states": list(self.dtmc.states),
                "absorbing_states": list(self.dtmc.absorbing_states),
                "transitions": self.dtmc.transitions,
                "absorption_probabilities": self.dtmc_solution.absorption_probabilities,
                "failure_probability": self.dtmc_solution.failure_probability,
                "success_probability": self.dtmc_solution.success_probability,
                "expected_steps": self.dtmc_solution.expected_steps,
            },
        }
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        return destination


def _canonical_component(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    match = re.fullmatch(r"joint_(\d+)", value)
    return f"j{match.group(1)}" if match else value


def _hazard_rate(probability: float, basis: str) -> float:
    seconds = {"per_second": 1.0, "per_minute": 60.0, "per_hour": 3600.0}
    if basis not in seconds:
        raise ValueError(
            "failure_probability_basis must be per_second, per_minute, or per_hour."
        )
    if probability >= 1.0:
        return math.inf
    return -math.log1p(-probability) / seconds[basis]


def _matching_usage(joint_summary: pd.DataFrame, skill_id: int, sources: tuple[str, ...]) -> pd.DataFrame:
    if joint_summary.empty or not sources:
        return joint_summary.iloc[0:0]
    rows = joint_summary[joint_summary["skill_id"].astype(int) == int(skill_id)].copy()
    canonical_sources = {_canonical_component(source) for source in sources}

    def matches(joint: object) -> bool:
        canonical = _canonical_component(str(joint))
        return canonical in canonical_sources or (
            "gripper" in canonical_sources and "gripper" in canonical
        )

    return rows[rows["joint"].map(matches)]


def analyze_reliability(
    behavior: BehavioralResult,
    config: RobotConfig,
    *,
    multipliers: ExposureMultipliers | None = None,
) -> ReliabilityResult:
    """Create and solve one component fault tree for every observed skill."""

    factors = multipliers or ExposureMultipliers()
    required_skill_columns = {"skill_id", "skill", "n_episodes", "total_duration"}
    if not required_skill_columns.issubset(behavior.skill_summary.columns):
        missing = sorted(required_skill_columns - set(behavior.skill_summary.columns))
        raise ValueError(f"Behavior skill summary is missing columns: {missing}")

    component_rows: list[dict[str, object]] = []
    fault_trees: dict[str, FaultTreeModel] = {}
    skill_rows: list[dict[str, object]] = []
    for skill_row in behavior.skill_summary.itertuples(index=False):
        skill_id = int(skill_row.skill_id)
        skill = str(skill_row.skill)
        episodes = max(1, int(skill_row.n_episodes))
        average_duration = float(skill_row.total_duration) / episodes
        probabilities: dict[str, float] = {}

        for name, component in config.components.items():
            usage = _matching_usage(behavior.joint_summary, skill_id, component.behavior_sources)
            velocity_times = tuple(
                float(usage.get(column, pd.Series(dtype=float)).sum()) / episodes
                for column in ("velocity_time_low", "velocity_time_medium", "velocity_time_high")
            )
            effort_times = tuple(
                float(usage.get(column, pd.Series(dtype=float)).sum()) / episodes
                for column in ("effort_time_low", "effort_time_medium", "effort_time_high")
            )
            average_active = (
                float(usage.get("total_active_time", pd.Series(dtype=float)).sum()) / episodes
                if not usage.empty
                else 0.0
            )

            if component.exposure == "skill_time":
                base_exposure = average_duration
                weighted_velocity_exposure = average_duration
            else:
                weighted_velocity_exposure = sum(
                    duration * multiplier
                    for duration, multiplier in zip(velocity_times, factors.velocity)
                )
                base_exposure = sum(velocity_times)
                if weighted_velocity_exposure == 0.0:
                    base_exposure = average_active
                    weighted_velocity_exposure = average_active

            effort_total = sum(effort_times)
            effort_factor = 1.0
            if effort_total > 0.0:
                effort_factor = sum(
                    duration * multiplier
                    for duration, multiplier in zip(effort_times, factors.effort)
                ) / effort_total
            effective_exposure = weighted_velocity_exposure * effort_factor
            rate = _hazard_rate(component.failure_probability, config.probability_basis)
            hazard = rate * effective_exposure
            probability = 1.0 if math.isinf(hazard) else -math.expm1(-hazard)
            probabilities[name] = probability
            component_rows.append({
                "skill_id": skill_id,
                "skill": skill,
                "component": name,
                "exposure_mode": component.exposure,
                "average_skill_duration": average_duration,
                "average_active_time": average_active,
                "velocity_time_low": velocity_times[0],
                "velocity_time_medium": velocity_times[1],
                "velocity_time_high": velocity_times[2],
                "base_exposure": base_exposure,
                "effort_factor": effort_factor,
                "effective_exposure": effective_exposure,
                "base_failure_probability": component.failure_probability,
                "probability_basis": config.probability_basis,
                "hazard_rate_per_second": rate,
                "hazard": hazard,
                "failure_probability": probability,
            })

        top_event = f"skill_{skill_id}_failure"
        tree = config.build_fault_tree(
            component_probabilities=probabilities,
            top_event=top_event,
        )
        fault_trees[skill] = tree
        bottom_up = bottom_up_probability(tree)
        bdd = bdd_probability(tree)
        skill_rows.append({
            "skill_id": skill_id,
            "skill": skill,
            "bottom_up_probability": bottom_up,
            "bdd_probability": bdd.probability,
            "bdd_nodes": bdd.node_count,
            "solver_difference": bottom_up - bdd.probability,
        })

    skill_probabilities = pd.DataFrame(skill_rows)
    dtmc = _build_dtmc(behavior.segments, skill_probabilities)
    return ReliabilityResult(
        component_failures=pd.DataFrame(component_rows),
        fault_trees=fault_trees,
        skill_probabilities=skill_probabilities,
        dtmc=dtmc,
        dtmc_solution=solve_dtmc(dtmc),
    )


def _build_dtmc(segments: pd.DataFrame, skill_probabilities: pd.DataFrame) -> DTMCModel:
    required = {"episode_key", "skill_id", "start_index"}
    if not required.issubset(segments.columns):
        missing = sorted(required - set(segments.columns))
        raise ValueError(f"Behavior segments are missing columns: {missing}")
    ordered_skills = [int(value) for value in skill_probabilities["skill_id"]]
    skill_states = {skill_id: f"skill_{skill_id}" for skill_id in ordered_skills}
    failure_states = {skill_id: f"skill_{skill_id}_failure" for skill_id in ordered_skills}
    start_counts = {skill_id: 0 for skill_id in ordered_skills}
    exits = {skill_id: {} for skill_id in ordered_skills}

    for _, episode in segments.sort_values(["episode_key", "start_index"]).groupby(
        "episode_key", sort=False
    ):
        sequence = [int(value) for value in episode["skill_id"]]
        if not sequence:
            continue
        start_counts[sequence[0]] += 1
        for index, skill_id in enumerate(sequence):
            destination = skill_states[sequence[index + 1]] if index + 1 < len(sequence) else "done"
            exits[skill_id][destination] = exits[skill_id].get(destination, 0) + 1

    episode_count = sum(start_counts.values())
    if episode_count == 0:
        raise ValueError("Behavior segments contain no episodes.")
    transitions: dict[str, dict[str, float]] = {
        "start": {
            skill_states[skill_id]: count / episode_count
            for skill_id, count in start_counts.items()
            if count
        }
    }
    probability_by_skill = {
        int(row.skill_id): float(row.bdd_probability)
        for row in skill_probabilities.itertuples(index=False)
    }
    for skill_id in ordered_skills:
        probability = probability_by_skill[skill_id]
        survival = 1.0 - probability
        counts = exits[skill_id] or {"done": 1}
        total = sum(counts.values())
        row = {failure_states[skill_id]: probability}
        for destination, count in counts.items():
            row[destination] = row.get(destination, 0.0) + survival * count / total
        transitions[skill_states[skill_id]] = row

    absorbing = (*failure_states.values(), "done")
    transitions.update({state: {state: 1.0} for state in absorbing})
    return DTMCModel(
        states=("start", *skill_states.values()),
        absorbing_states=absorbing,
        transitions=transitions,
    )
