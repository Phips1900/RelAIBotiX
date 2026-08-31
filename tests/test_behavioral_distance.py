import numpy as np
import pandas as pd
import pytest

from relaibotix.behavioral.behavioral_analysis_v2 import BehavioralAnalyzer


def test_component_inference_supports_numbered_and_named_joints():
    analyzer = BehavioralAnalyzer()

    columns = analyzer._infer_component_cols([
        "joint_pos_1",
        "joint_vel_1",
        "joint_pos_joint_left_wheel",
        "joint_vel_joint_left_wheel",
        "joint_effort_joint_left_wheel",
    ])

    assert columns["j1"] == {"pos": 0, "vel": 1}
    assert columns["joint_left_wheel"] == {"pos": 2, "vel": 3, "eff": 4}


def test_joint_distance_is_position_path_length():
    analyzer = BehavioralAnalyzer()
    features = pd.DataFrame({"joint_pos_1": [0.0, 1.0, 0.25, 0.5]})

    metrics = analyzer._component_metrics(
        features,
        {"j1": {"pos": 0}},
        np.array([0.0, 0.1, 0.4, 1.0]),
        0,
        3,
    )

    assert metrics["j1"].metrics["j1.pos_travel_distance"] == pytest.approx(2.0)


def test_joint_distance_does_not_bridge_missing_samples():
    analyzer = BehavioralAnalyzer()
    features = pd.DataFrame({"joint_pos_1": [0.0, 1.0, np.nan, 4.0, 5.5]})

    metrics = analyzer._component_metrics(
        features,
        {"j1": {"pos": 0}},
        np.arange(5, dtype=float),
        0,
        4,
    )

    assert metrics["j1"].metrics["j1.pos_travel_distance"] == pytest.approx(2.5)


def test_summary_reports_distance_by_skill_and_joint():
    analyzer = BehavioralAnalyzer()
    features = pd.DataFrame({
        "joint_pos_1": [0.0, 1.0, 0.25, 0.5],
        "joint_vel_1": [0.0, 1.0, -0.75, 0.25],
    })
    traces = analyzer.analyze(
        features=features,
        feature_names=list(features.columns),
        labels=np.zeros(4, dtype=int),
        timestamps=np.array([0.0, 0.1, 0.4, 1.0]),
        episode_labels=np.zeros(4, dtype=int),
    )

    distance = analyzer.summarize(traces)["joint_distance"]

    assert distance.to_dict(orient="records") == [{
        "skill": "Move",
        "component": "j1",
        "total_distance": pytest.approx(2.0),
        "avg_distance_per_episode": pytest.approx(2.0),
        "max_distance_per_episode": pytest.approx(2.0),
        "n_episodes": 1,
    }]
