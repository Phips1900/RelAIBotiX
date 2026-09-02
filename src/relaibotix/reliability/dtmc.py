"""Small validated DTMC model and numerical absorption solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DTMCModel:
    states: tuple[str, ...]
    absorbing_states: tuple[str, ...]
    transitions: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        transitions = {
            str(source): {str(target): float(value) for target, value in row.items()}
            for source, row in self.transitions.items()
        }
        object.__setattr__(self, "transitions", transitions)
        self.validate()

    def validate(self) -> None:
        order = (*self.states, *self.absorbing_states)
        if not self.states:
            raise ValueError("DTMC requires at least one transient state.")
        if len(set(order)) != len(order):
            raise ValueError("DTMC state names must be unique.")
        known = set(order)
        for state in order:
            if state not in self.transitions:
                raise ValueError(f"DTMC has no transition row for '{state}'.")
            row = self.transitions[state]
            unknown = set(row) - known
            if unknown:
                raise ValueError(f"DTMC row '{state}' references unknown states: {sorted(unknown)}")
            if any(not 0.0 <= probability <= 1.0 for probability in row.values()):
                raise ValueError(f"DTMC row '{state}' contains a probability outside [0, 1].")
            if not np.isclose(sum(row.values()), 1.0, atol=1e-10):
                raise ValueError(f"DTMC row '{state}' must sum to one.")
        for state in self.absorbing_states:
            if self.transitions[state] != {state: 1.0}:
                raise ValueError(f"Absorbing state '{state}' must have a unit self-loop.")

    def get_states(self) -> list[str]:
        return list(self.states)

    def get_absorbing_states(self) -> list[str]:
        return list(self.absorbing_states)

    def get_transitions(self) -> dict[str, dict[str, float]]:
        return {source: dict(row) for source, row in self.transitions.items()}

    def matrix(self) -> np.ndarray:
        order = (*self.states, *self.absorbing_states)
        index = {state: position for position, state in enumerate(order)}
        matrix = np.zeros((len(order), len(order)), dtype=float)
        for source, row in self.transitions.items():
            for target, probability in row.items():
                matrix[index[source], index[target]] = probability
        return matrix


@dataclass(frozen=True)
class DTMCSolution:
    absorption_probabilities: Mapping[str, float]
    expected_steps: float

    @property
    def failure_probability(self) -> float:
        return float(sum(
            probability
            for state, probability in self.absorption_probabilities.items()
            if state.endswith("_failure")
        ))

    @property
    def completion_without_modeled_failure_probability(self) -> float:
        """Probability of reaching ``done``; this is not task-success detection."""

        return float(self.absorption_probabilities.get("done", 0.0))


def solve_dtmc(model: DTMCModel, *, start_state: str = "start") -> DTMCSolution:
    """Solve absorption probabilities and expected steps from one start state."""

    if start_state not in model.states:
        raise ValueError(f"Unknown DTMC start state: {start_state}")
    transient_count = len(model.states)
    matrix = model.matrix()
    q = matrix[:transient_count, :transient_count]
    r = matrix[:transient_count, transient_count:]
    fundamental = np.linalg.solve(np.eye(transient_count) - q, np.eye(transient_count))
    absorption = fundamental @ r
    expected_steps = fundamental @ np.ones((transient_count, 1))
    start = model.states.index(start_state)
    return DTMCSolution(
        absorption_probabilities={
            state: float(absorption[start, index])
            for index, state in enumerate(model.absorbing_states)
        },
        expected_steps=float(expected_steps[start, 0]),
    )
