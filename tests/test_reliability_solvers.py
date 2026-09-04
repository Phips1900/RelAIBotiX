import json

import numpy as np
import pytest

from relaibotix.reliability import (
    FaultTree,
    FaultTreeModel,
    Gate,
    bdd_probability,
    bottom_up_probability,
    load_robot_config,
    analyze_component_sensitivity,
    analyze_reliability,
)
from relaibotix.behavioral.results import BehavioralResult
import pandas as pd
from relaibotix.reliability.solver import create_mc_transition_matrix, solve_mc
from relaibotix.reliability.graph import create_mc_graph
from relaibotix.reliability.storm import run_storm
from types import SimpleNamespace


EXPOSURE_ASSUMPTIONS = {
    "source": "test_expert",
    "position_step": 0.001,
    "velocity_active": 0.03,
    "effort_active": 0.1,
    "velocity_bands": [0.5, 1.0],
    "effort_bands": [0.2, 0.6],
    "velocity_multipliers": [1.0, 2.0, 5.0],
    "effort_multipliers": [1.0, 1.25, 1.75],
    "distance_multipliers": [1.0, 1.5, 2.0],
}


def _write_robot_config(path, components, *, assumptions=None):
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "robot": {"id": "test_robot", "name": "Test Robot", "type": "test"},
        "failure_probability_basis": "per_minute",
        "failure_probability_source": "test_source",
        "exposure_assumptions": assumptions or EXPOSURE_ASSUMPTIONS,
        "components": components,
    }))


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


def test_existing_robot_config_defines_measured_components_and_redundancy():
    config = load_robot_config("configs/robots/franka.json")

    assert config.robot_id == "franka_emika_panda"
    assert config.name == "Franka Emika Panda"
    assert config.measured_component_count == 8
    assert config.redundant_components == {
        "controller": 2,
        "power_supply": 2,
    }

    tree = config.build_fault_tree(active_components=["joint_1", "controller"])
    assert tree.gates["loss_of_controller"] == Gate(
        "AND", ("controller_1", "controller_2")
    )
    assert bottom_up_probability(tree) == pytest.approx(bdd_probability(tree).probability)


def test_libero_config_matches_position_velocity_only_panda_logs():
    config = load_robot_config("configs/robots/franka_libero.json")

    assert config.robot_id == "franka_emika_panda_libero"
    assert config.measured_component_count == 8
    assert config.components["joint_1"].features["position"] == ("joint_pos_1",)
    assert config.components["joint_1"].features["velocity"] == ("joint_vel_1",)
    assert config.components["joint_1"].features["effort"] == ()
    assert config.components["camera"].redundancy_copies == 1


def test_hello_stretch_config_maps_logged_components():
    config = load_robot_config("configs/robots/hello_stretch.json")

    assert config.robot_id == "hello_stretch_3"
    assert config.robot_type == "mobile_manipulator"
    assert config.measured_component_count == 7
    assert config.redundant_components == {}
    assert config.components["telescoping_arm"].features["position"] == (
        "joint_pos_joint_arm_l3",
        "joint_pos_joint_arm_l2",
        "joint_pos_joint_arm_l1",
        "joint_pos_joint_arm_l0",
    )
    assert config.components["wrist"].features["position"] == (
        "joint_pos_joint_wrist_yaw",
        "joint_pos_joint_wrist_pitch",
        "joint_pos_joint_wrist_roll",
    )
    assert config.components["head"].features["position"] == (
        "joint_pos_joint_head_pan",
        "joint_pos_joint_head_tilt",
    )
    assert "wrist_yaw" not in config.components
    assert "head_pan" not in config.components


def test_legacy_robot_config_shape_is_rejected(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"robot":"Legacy Robot","robot_type":"Manipulator",'
        '"components":{"Joint_1":{"failure_probability":0.1,"redundancy":false}}}'
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_robot_config(path)


def test_structured_redundancy_can_define_copy_count(tmp_path):
    path = tmp_path / "robot.json"
    _write_robot_config(path, {
        "drive": {
            "type": "drive",
            "always_active": True,
            "failure_probability": 0.1,
            "redundancy": {"copies": 3, "mode": "parallel"},
        }
    })

    config = load_robot_config(path)
    tree = config.build_fault_tree()

    assert config.redundant_components == {"drive": 3}
    assert tree.gates["loss_of_drive"].children == ("drive_1", "drive_2", "drive_3")
    assert bdd_probability(tree).probability == pytest.approx(0.001)


def test_expert_exposure_assumptions_are_loaded(tmp_path):
    path = tmp_path / "robot.json"
    assumptions = dict(EXPOSURE_ASSUMPTIONS)
    assumptions.update({
        "source": "expert-review-1",
        "velocity_bands": [0.2, 0.7],
        "velocity_multipliers": [1.0, 1.5, 2.0],
        "effort_multipliers": [1.0, 1.5, 2.0],
    })
    _write_robot_config(path, {
        "drive": {
            "type": "drive",
            "always_active": True,
            "failure_probability": 0.1,
            "redundancy": {"copies": 1, "mode": "parallel"},
        }
    }, assumptions=assumptions)

    assumptions = load_robot_config(path).exposure_assumptions

    assert assumptions.source == "expert-review-1"
    assert assumptions.velocity_bands == pytest.approx((0.2, 0.7))
    assert assumptions.distance_multipliers == pytest.approx((1.0, 1.5, 2.0))


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
            "n_segments": 2,
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
            "total_traveled_distance": 4.0,
        }]),
    )
    config_path = tmp_path / "robot.json"
    _write_robot_config(config_path, {
        "joint_1": {
            "type": "revolute_joint",
            "features": {"position": "joint_pos_1", "velocity": "joint_vel_1"},
            "failure_probability": 0.01,
            "redundancy": {"copies": 1, "mode": "parallel"},
            "distance_thresholds": [1.0, 3.0],
            "distance_unit": "radian",
        },
        "controller": {
            "type": "controller",
            "always_active": True,
            "failure_probability": 0.02,
            "redundancy": {"copies": 1, "mode": "parallel"},
        },
    })

    result = analyze_reliability(behavior, load_robot_config(config_path))
    rows = result.component_failures.set_index("component")

    assert rows.loc["joint_1", "base_exposure"] == pytest.approx(3.0)
    assert rows.loc["joint_1", "average_traveled_distance"] == pytest.approx(2.0)
    assert rows.loc["joint_1", "distance_band"] == "medium"
    assert rows.loc["joint_1", "distance_factor"] == pytest.approx(1.5)
    assert rows.loc["joint_1", "effective_exposure"] == pytest.approx(16.5)
    assert rows.loc["controller", "effective_exposure"] == pytest.approx(10.0)
    assert result.skill_probabilities.iloc[0]["bottom_up_probability"] == pytest.approx(
        result.skill_probabilities.iloc[0]["bdd_probability"]
    )
    assert (
        result.dtmc_solution.failure_probability
        + result.dtmc_solution.completion_without_modeled_failure_probability
    ) == pytest.approx(1.0)
    assert result.dtmc_solution.failure_probability == pytest.approx(
        result.skill_probabilities.iloc[0]["bdd_probability"]
    )

    sensitivity = analyze_component_sensitivity(
        behavior,
        load_robot_config(config_path),
        baseline=result,
    )
    assert sensitivity["influence_rank"].tolist() == [1, 2]
    assert (sensitivity["requested_factor"] == 10.0).all()
    assert (
        sensitivity["perturbed_system_failure_probability"]
        >= sensitivity["baseline_system_failure_probability"]
    ).all()
    assert sensitivity["absolute_system_probability_change"].is_monotonic_decreasing


def test_sensitivity_factor_is_validated(tmp_path):
    config_path = tmp_path / "robot.json"
    _write_robot_config(config_path, {
        "controller": {
            "type": "controller",
            "always_active": True,
            "failure_probability": 0.1,
            "redundancy": {"copies": 1, "mode": "parallel"},
        }
    })
    behavior = BehavioralResult(
        segments=pd.DataFrame([{"episode_key": "demo_0", "skill_id": 1, "start_index": 0}]),
        joint_metrics=pd.DataFrame(),
        skill_summary=pd.DataFrame([{
            "skill_id": 1,
            "skill": "move",
            "n_segments": 1,
            "total_duration": 1.0,
        }]),
        joint_summary=pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="Sensitivity factor"):
        analyze_component_sensitivity(behavior, load_robot_config(config_path), factor=1.0)


def test_distance_thresholds_are_validated(tmp_path):
    path = tmp_path / "robot.json"
    _write_robot_config(path, {
        "joint_1": {
            "type": "revolute_joint",
            "features": {"position": "joint_pos_1"},
            "failure_probability": 0.1,
            "redundancy": {"copies": 1, "mode": "parallel"},
            "distance_thresholds": [2.0, 1.0],
            "distance_unit": "radian",
        }
    })

    with pytest.raises(ValueError, match="Distance thresholds"):
        load_robot_config(path)


def test_reliability_rejects_behavior_threshold_mismatch(tmp_path):
    behavior = BehavioralResult(
        segments=pd.DataFrame([{"episode_key": "demo_0", "skill_id": 1, "start_index": 0}]),
        joint_metrics=pd.DataFrame(),
        skill_summary=pd.DataFrame([{
            "skill_id": 1,
            "skill": "move",
            "n_segments": 1,
            "total_duration": 1.0,
        }]),
        joint_summary=pd.DataFrame(),
        metadata={"behavioral_thresholds": {
            "position_step": 0.001,
            "velocity_active": 0.03,
            "effort_active": 0.1,
            "velocity_bands": [9.0, 10.0],
            "effort_bands": [0.2, 0.6],
        }},
    )
    config_path = tmp_path / "robot.json"
    _write_robot_config(config_path, {
        "controller": {
            "type": "controller",
            "always_active": True,
            "failure_probability": 0.1,
            "redundancy": {"copies": 1, "mode": "parallel"},
        }
    })

    with pytest.raises(ValueError, match="different exposure thresholds"):
        analyze_reliability(behavior, load_robot_config(config_path))


def test_storm_backend_parses_one_result_per_property(tmp_path, monkeypatch):
    model = tmp_path / "model.pm"
    properties = tmp_path / "model.pctl"
    model.write_text("dtmc\n")
    properties.write_text('P=? [ F "failure" ]\nP=? [ F "done" ]\n')
    captured = {"commands": []}

    monkeypatch.setattr("relaibotix.reliability.storm.shutil.which", lambda executable: "/usr/bin/storm")

    def fake_run(command, **options):
        captured["commands"].append(command)
        captured["options"] = options
        value = "1/8" if len(captured["commands"]) == 1 else "7/8"
        return SimpleNamespace(
            returncode=0,
            stdout=f"Result (for initial states): {value}\n",
            stderr="",
        )

    monkeypatch.setattr("relaibotix.reliability.storm.subprocess.run", fake_run)
    result = run_storm(model, properties, exact=True)

    assert result.values == (0.125, 0.875)
    assert captured["commands"] == [
        ["/usr/bin/storm", "--prism", str(model), "--prop", 'P=? [ F "failure" ]', "--exact"],
        ["/usr/bin/storm", "--prism", str(model), "--prop", 'P=? [ F "done" ]', "--exact"],
    ]
    assert captured["options"]["timeout"] == 120.0
