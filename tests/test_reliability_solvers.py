import numpy as np
import pytest

from relaibotix.reliability import (
    FaultTree,
    FaultTreeModel,
    Gate,
    bdd_probability,
    bottom_up_probability,
    load_robot_config,
)
from relaibotix.reliability.solver import create_mc_transition_matrix, solve_mc
from relaibotix.reliability.graph import create_mc_graph


def test_bottom_up_and_bdd_agree_for_a_tree():
    model = FaultTreeModel(
        top_event="system_failure",
        basic_events={"motor": 0.1, "camera": 0.2},
        gates={"system_failure": Gate("OR", ("motor", "camera"))},
    )

    assert bottom_up_probability(model) == pytest.approx(0.28)
    result = bdd_probability(model)
    assert result.probability == pytest.approx(0.28)
    assert result.variable_order == ("camera", "motor")


def test_bdd_handles_a_shared_basic_event_exactly():
    model = FaultTreeModel(
        top_event="system_failure",
        basic_events={"a": 0.1, "b": 0.2, "c": 0.3},
        gates={
            "ab": Gate("AND", ("a", "b")),
            "ac": Gate("AND", ("a", "c")),
            "system_failure": Gate("OR", ("ab", "ac")),
        },
    )

    assert bottom_up_probability(model) == pytest.approx(0.0494)
    assert bdd_probability(model).probability == pytest.approx(0.044)


def test_fault_tree_legacy_class_exposes_both_solvers():
    fault_tree = FaultTree("example", top_event="failure")
    fault_tree.add_basic_events({"a": 0.1, "b": 0.2})
    fault_tree.add_single_gate("failure", "AND", ["a", "b"])

    assert fault_tree.evaluate() == pytest.approx(0.02)
    assert fault_tree.evaluate_bdd().probability == pytest.approx(0.02)


def test_fault_tree_validation_rejects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        FaultTreeModel(
            top_event="top",
            basic_events={"a": 0.1},
            gates={
                "top": Gate("OR", ("loop", "a")),
                "loop": Gate("AND", ("top", "a")),
            },
        )


class _SmallMarkovChain:
    def __init__(self):
        self.states = ["run"]
        self.absorbing = ["failed", "done"]
        self.transitions = {
            "run": {"failed": 0.2, "done": 0.8},
            "failed": {"failed": 1.0},
            "done": {"done": 1.0},
        }

    def get_states(self):
        return self.states

    def get_absorbing_states(self):
        return self.absorbing

    def get_transitions(self):
        return self.transitions


def test_markov_solver_does_not_mutate_state_lists():
    chain = _SmallMarkovChain()
    matrix = create_mc_transition_matrix(chain.states, chain.absorbing, chain.transitions)
    absorption, time = solve_mc(chain)

    assert chain.states == ["run"]
    assert chain.absorbing == ["failed", "done"]
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(absorption, [[0.2, 0.8]])
    np.testing.assert_allclose(time, [[1.0]])

    create_mc_graph(chain)
    assert chain.states == ["run"]


def test_existing_robot_config_defines_joints_and_redundancy():
    config = load_robot_config("config_files/robots/franka_config.json")

    assert config.name == "Franka Emika Panda"
    assert config.joint_count == 7
    assert config.redundant_components == {
        "Controller": 2,
        "Power_Supply": 2,
        "Sensors": 2,
    }

    tree = config.build_fault_tree(active_components=["Joint_1", "Controller"])
    assert tree.gates["loss_of_Controller"] == Gate(
        "AND", ("Controller_1", "Controller_2")
    )
    assert bottom_up_probability(tree) == pytest.approx(bdd_probability(tree).probability)


def test_structured_redundancy_can_define_copy_count(tmp_path):
    path = tmp_path / "robot.json"
    path.write_text(
        '{"robot":"test","robot_type":"mobile","components":{'
        '"drive":{"failure_probability":0.1,"redundancy":{"copies":3}}}}'
    )

    config = load_robot_config(path)
    tree = config.build_fault_tree()

    assert config.redundant_components == {"drive": 3}
    assert tree.gates["loss_of_drive"].children == ("drive_1", "drive_2", "drive_3")
    assert bdd_probability(tree).probability == pytest.approx(0.001)
