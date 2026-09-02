import numpy as np
import pytest

from relaibotix.reliability import (
    FaultTree,
    FaultTreeModel,
    Gate,
    bdd_probability,
    bottom_up_probability,
    load_robot_config,
    analyze_reliability,
)
from relaibotix.behavioral.results import BehavioralResult
import pandas as pd
from relaibotix.reliability.solver import create_mc_transition_matrix, solve_mc
from relaibotix.reliability.graph import create_mc_graph
from relaibotix.reliability.storm import run_storm
from types import SimpleNamespace


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


def test_behavior_exposure_builds_auditable_per_skill_fault_tree(tmp_path):
    behavior = BehavioralResult(
        segments=pd.DataFrame([
            {"episode_key": "demo_0", "skill_id": 1, "start_index": 0},
            {"episode_key": "demo_1", "skill_id": 1, "start_index": 1},
        ]),
        joint_metrics=pd.DataFrame(),
        skill_summary=pd.DataFrame([{
            "skill_id": 1,
            "skill": "move",
            "n_episodes": 2,
            "total_duration": 20.0,
        }]),
        joint_summary=pd.DataFrame([{
            "skill_id": 1,
            "skill": "move",
            "joint": "j1",
            "total_active_time": 8.0,
            "velocity_time_low": 2.0,
            "velocity_time_medium": 2.0,
            "velocity_time_high": 2.0,
            "effort_time_low": 2.0,
            "effort_time_medium": 0.0,
            "effort_time_high": 2.0,
        }]),
    )
    config_path = tmp_path / "robot.json"
    config_path.write_text(
        '{"robot":"test","robot_type":"arm","components":{'
        '"Joint_1":{"failure_probability":0.01,"redundancy":false},'
        '"Controller":{"failure_probability":0.02,"redundancy":false}}}'
    )

    result = analyze_reliability(behavior, load_robot_config(config_path))
    rows = result.component_failures.set_index("component")

    assert rows.loc["Joint_1", "base_exposure"] == pytest.approx(3.0)
    assert rows.loc["Joint_1", "effective_exposure"] == pytest.approx(11.0)
    assert rows.loc["Controller", "effective_exposure"] == pytest.approx(10.0)
    assert result.skill_probabilities.iloc[0]["bottom_up_probability"] == pytest.approx(
        result.skill_probabilities.iloc[0]["bdd_probability"]
    )
    assert result.dtmc_solution.failure_probability + result.dtmc_solution.success_probability == pytest.approx(1.0)
    assert result.dtmc_solution.failure_probability == pytest.approx(
        result.skill_probabilities.iloc[0]["bdd_probability"]
    )


def test_storm_backend_parses_one_result_per_property(tmp_path, monkeypatch):
    model = tmp_path / "model.pm"
    properties = tmp_path / "model.pctl"
    model.write_text("dtmc\n")
    properties.write_text('P=? [ F "failure" ]\nP=? [ F "done" ]\n')
    captured = {}

    monkeypatch.setattr("relaibotix.reliability.storm.shutil.which", lambda executable: "/usr/bin/storm")

    def fake_run(command, **options):
        captured["command"] = command
        captured["options"] = options
        return SimpleNamespace(
            returncode=0,
            stdout="Result (initial states): 1/8\nResult (initial states): 7/8\n",
            stderr="",
        )

    monkeypatch.setattr("relaibotix.reliability.storm.subprocess.run", fake_run)
    result = run_storm(model, properties, exact=True)

    assert result.values == (0.125, 0.875)
    assert captured["command"] == [
        "/usr/bin/storm", "--prism", str(model), "--prop", str(properties), "--exact"
    ]
    assert captured["options"]["timeout"] == 120.0
