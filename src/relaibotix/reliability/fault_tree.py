"""Validated fault-tree representation and exact probability solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Gate:
    """An AND or OR gate with named child events."""

    operator: str
    children: tuple[str, ...]

    def __post_init__(self) -> None:
        operator = self.operator.upper()
        if operator not in {"AND", "OR"}:
            raise ValueError(f"Unsupported gate operator: {self.operator}")
        if not self.children:
            raise ValueError("Fault-tree gates must have at least one child.")
        object.__setattr__(self, "operator", operator)


@dataclass(frozen=True)
class FaultTreeModel:
    """Solver-neutral fault tree with independent basic-event probabilities."""

    top_event: str
    basic_events: Mapping[str, float]
    gates: Mapping[str, Gate]

    def __post_init__(self) -> None:
        basics = {str(name): float(probability) for name, probability in self.basic_events.items()}
        gates = {str(name): gate for name, gate in self.gates.items()}
        object.__setattr__(self, "basic_events", basics)
        object.__setattr__(self, "gates", gates)
        self.validate()

    @classmethod
    def from_legacy(cls, fault_tree) -> "FaultTreeModel":
        gates: dict[str, Gate] = {}
        for name, definition in fault_tree.get_gates().items():
            if len(definition) != 1:
                raise ValueError(f"Gate '{name}' must define exactly one operator.")
            operator, children = next(iter(definition.items()))
            gates[str(name)] = Gate(str(operator), tuple(map(str, children)))
        return cls(
            top_event=str(fault_tree.get_top_event()),
            basic_events=fault_tree.get_basic_events(),
            gates=gates,
        )

    def validate(self) -> None:
        if not self.top_event:
            raise ValueError("Fault tree requires a top event.")
        if self.top_event not in self.gates:
            raise ValueError(f"Top event '{self.top_event}' is not a gate.")
        overlap = set(self.basic_events) & set(self.gates)
        if overlap:
            raise ValueError(f"Events cannot be both basic events and gates: {sorted(overlap)}")
        invalid = {
            name: probability
            for name, probability in self.basic_events.items()
            if not 0.0 <= probability <= 1.0
        }
        if invalid:
            raise ValueError(f"Basic-event probabilities must lie in [0, 1]: {invalid}")
        known = set(self.basic_events) | set(self.gates)
        for name, gate in self.gates.items():
            missing = set(gate.children) - known
            if missing:
                raise ValueError(f"Gate '{name}' references unknown events: {sorted(missing)}")
        self.bottom_up_order()

    def bottom_up_order(self) -> tuple[str, ...]:
        """Return gates from leaf-most to top without recursive traversal."""

        unresolved = {
            name: sum(child in self.gates for child in gate.children)
            for name, gate in self.gates.items()
        }
        dependents: dict[str, list[str]] = {name: [] for name in self.gates}
        for parent, gate in self.gates.items():
            for child in gate.children:
                if child in self.gates:
                    dependents[child].append(parent)

        ready = sorted(name for name, count in unresolved.items() if count == 0)
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for parent in sorted(dependents[name]):
                unresolved[parent] -= 1
                if unresolved[parent] == 0:
                    ready.append(parent)
            ready.sort()
        if len(order) != len(self.gates):
            cyclic = sorted(name for name, count in unresolved.items() if count > 0)
            raise ValueError(f"Fault tree contains a gate cycle: {cyclic}")
        return tuple(order)


def bottom_up_probability(model: FaultTreeModel) -> float:
    """Evaluate gates bottom-up using the traditional independence equations."""

    probabilities = dict(model.basic_events)
    for name in model.bottom_up_order():
        gate = model.gates[name]
        children = [probabilities[child] for child in gate.children]
        if gate.operator == "AND":
            probability = 1.0
            for child in children:
                probability *= child
        else:
            complement = 1.0
            for child in children:
                complement *= 1.0 - child
            probability = 1.0 - complement
        probabilities[name] = probability
    return float(probabilities[model.top_event])


@dataclass(frozen=True)
class BDDResult:
    probability: float
    node_count: int
    variable_order: tuple[str, ...]


class _BDDManager:
    FALSE = 0
    TRUE = 1

    def __init__(self, variables: Iterable[str]):
        self.variables = tuple(variables)
        self.variable_index = {name: index for index, name in enumerate(self.variables)}
        self.nodes: dict[int, tuple[int, int, int]] = {}
        self.unique: dict[tuple[int, int, int], int] = {}
        self.next_id = 2
        self.apply_cache: dict[tuple[str, int, int], int] = {}

    def make(self, variable: int, low: int, high: int) -> int:
        if low == high:
            return low
        key = (variable, low, high)
        if key not in self.unique:
            self.unique[key] = self.next_id
            self.nodes[self.next_id] = key
            self.next_id += 1
        return self.unique[key]

    def variable(self, name: str) -> int:
        return self.make(self.variable_index[name], self.FALSE, self.TRUE)

    def apply(self, operator: str, left: int, right: int) -> int:
        if operator in {"AND", "OR"} and right < left:
            left, right = right, left
        key = (operator, left, right)
        if key in self.apply_cache:
            return self.apply_cache[key]
        if operator == "AND":
            if left == self.FALSE or right == self.FALSE:
                return self.FALSE
            if left == self.TRUE:
                return right
            if right == self.TRUE or left == right:
                return left
        elif operator == "OR":
            if left == self.TRUE or right == self.TRUE:
                return self.TRUE
            if left == self.FALSE:
                return right
            if right == self.FALSE or left == right:
                return left
        else:
            raise ValueError(f"Unsupported BDD operation: {operator}")

        left_node = self.nodes[left]
        right_node = self.nodes[right]
        variable = min(left_node[0], right_node[0])
        left_low, left_high = (left_node[1], left_node[2]) if left_node[0] == variable else (left, left)
        right_low, right_high = (right_node[1], right_node[2]) if right_node[0] == variable else (right, right)
        low = self.apply(operator, left_low, right_low)
        high = self.apply(operator, left_high, right_high)
        result = self.make(variable, low, high)
        self.apply_cache[key] = result
        return result

    def probability(self, root: int, probabilities: Mapping[str, float]) -> float:
        values: dict[int, float] = {self.FALSE: 0.0, self.TRUE: 1.0}
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if node in values:
                continue
            variable, low, high = self.nodes[node]
            if expanded:
                probability = float(probabilities[self.variables[variable]])
                values[node] = (1.0 - probability) * values[low] + probability * values[high]
            else:
                stack.append((node, True))
                stack.append((high, False))
                stack.append((low, False))
        return values[root]

    def reachable_node_count(self, root: int) -> int:
        reachable = {root}
        pending = [root]
        while pending:
            node = pending.pop()
            if node in {self.FALSE, self.TRUE}:
                continue
            _, low, high = self.nodes[node]
            for child in (low, high):
                if child not in reachable:
                    reachable.add(child)
                    pending.append(child)
        return len(reachable)


def bdd_probability(
    model: FaultTreeModel,
    *,
    variable_order: Iterable[str] | None = None,
) -> BDDResult:
    """Compile the fault tree to a reduced ordered BDD and evaluate it exactly."""

    order = tuple(variable_order) if variable_order is not None else tuple(sorted(model.basic_events))
    if set(order) != set(model.basic_events) or len(order) != len(model.basic_events):
        raise ValueError("BDD variable order must contain every basic event exactly once.")
    manager = _BDDManager(order)
    roots = {name: manager.variable(name) for name in order}
    for name in model.bottom_up_order():
        gate = model.gates[name]
        identity = manager.TRUE if gate.operator == "AND" else manager.FALSE
        root = identity
        for child in gate.children:
            root = manager.apply(gate.operator, root, roots[child])
        roots[name] = root
    top = roots[model.top_event]
    return BDDResult(
        probability=manager.probability(top, model.basic_events),
        node_count=manager.reachable_node_count(top),
        variable_order=order,
    )
