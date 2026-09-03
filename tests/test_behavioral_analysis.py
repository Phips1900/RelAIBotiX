import json

import h5py
import numpy as np
import pytest

from relaibotix.behavioral import BehavioralAnalyzer


FEATURE_NAMES = [
    "joint_pos_1",
    "joint_vel_1",
    "joint_effort_1",
    "joint_pos_joint_left_wheel",
    "joint_vel_joint_left_wheel",
]


def sample_data():
    features = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.4, 0.1, 0.5, 0.2],
        [0.5, -0.8, 0.5, 1.0, 0.7],
        [0.0, -0.4, 0.8, 0.0, -0.7],
        [0.5, 0.4, 0.2, 1.0, 0.4],
        [1.0, 0.4, 0.2, 2.0, 0.4],
    ])
    return features, np.array([0, 0, 1, 1, 1, 1]), np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]), np.array([0, 0, 0, 1, 1, 1])


def test_analysis_uses_explicit_episode_and_skill_boundaries():
    features, labels, timestamps, episodes = sample_data()
    result = BehavioralAnalyzer(skill_names={0: "Move", 1: "Pick"}).analyze(
        features=features,
        feature_names=FEATURE_NAMES,
        skill_labels=labels,
        timestamps=timestamps,
        episode_ids=episodes,
    )

    assert result.segments[["episode_id", "skill", "samples"]].to_dict(orient="records") == [
        {"episode_id": 0, "skill": "Move", "samples": 2},
        {"episode_id": 0, "skill": "Pick", "samples": 1},
        {"episode_id": 1, "skill": "Pick", "samples": 3},
    ]
    assert "success" not in result.segments.columns


def test_analysis_reports_numbered_and_named_joint_distance():
    features, labels, timestamps, episodes = sample_data()
    result = BehavioralAnalyzer().analyze(
        features=features,
        feature_names=FEATURE_NAMES,
        skill_labels=labels,
        timestamps=timestamps,
        episode_ids=episodes,
    )

    episode_one = result.joint_metrics[
        (result.joint_metrics["episode_id"] == 1) & (result.joint_metrics["skill_id"] == 1)
    ].set_index("joint")
    assert episode_one.loc["j1", "traveled_distance"] == pytest.approx(1.0)
    assert episode_one.loc["joint_left_wheel", "traveled_distance"] == pytest.approx(2.0)


def test_analysis_recognizes_actuator_force_channels():
    features = np.array([[0.0, 0.0], [1.0, 0.4], [2.0, -0.8]])
    result = BehavioralAnalyzer().analyze(
        features=features,
        feature_names=["joint_pos_joint_lift", "actuator_force_lift"],
        skill_labels=np.array([1, 1, 1]),
        timestamps=np.array([0.0, 0.5, 1.0]),
        episode_ids=np.array([0, 0, 0]),
    )

    lift = result.joint_metrics.iloc[0]
    assert lift["joint"] == "joint_lift"
    assert lift["traveled_distance"] == pytest.approx(2.0)
    assert lift["max_abs_effort"] == pytest.approx(0.8)


def test_configured_multi_axis_component_counts_time_once():
    features = np.array([
        [0.0, 0.0, 0.2, 0.8, 0.1, 0.4],
        [1.0, 2.0, 0.2, 0.8, 0.1, 0.4],
        [2.0, 0.0, 0.2, 0.8, 0.1, 0.4],
    ])
    names = ["p1", "p2", "v1", "v2", "e1", "e2"]
    result = BehavioralAnalyzer(component_features={
        "head": {
            "position": ("p1", "p2"),
            "velocity": ("v1", "v2"),
            "effort": ("e1", "e2"),
        }
    }).analyze(
        features=features,
        feature_names=names,
        skill_labels=np.array([1, 1, 1]),
        timestamps=np.array([0.0, 1.0, 2.0]),
        episode_ids=np.array([0, 0, 0]),
    )

    head = result.joint_metrics.iloc[0]
    assert head["joint"] == "head"
    assert head["position_axes"] == 2
    assert head["traveled_distance"] == pytest.approx(6.0)
    assert head["velocity_time_medium"] == pytest.approx(2.0)
    assert head["effort_time_medium"] == pytest.approx(2.0)
    assert head["active_time"] == pytest.approx(2.0)


def test_analysis_requires_detector_labels():
    features, labels, timestamps, episodes = sample_data()
    labels[0] = -1
    with pytest.raises(ValueError, match="detector-produced"):
        BehavioralAnalyzer().analyze(
            features=features,
            feature_names=FEATURE_NAMES,
            skill_labels=labels,
            timestamps=timestamps,
            episode_ids=episodes,
        )


def test_analyze_h5_and_export(tmp_path):
    features, labels, timestamps, episodes = sample_data()
    input_path = tmp_path / "input.h5"
    with h5py.File(input_path, "w") as output:
        dataset = output.create_dataset("features", data=features)
        dataset.attrs["feature_names"] = FEATURE_NAMES
        output.create_dataset("timestamps", data=timestamps)
        output.create_dataset("episode_ids", data=episodes)
        skills = output.create_group("skills")
        skills.create_dataset("predicted", data=labels)

    result = BehavioralAnalyzer().analyze_h5(input_path)
    csv_directory = result.write_csv(tmp_path / "csv")
    json_path = result.write_json(tmp_path / "behavior.json")

    assert (csv_directory / "joint_metrics.csv").is_file()
    assert set(json.loads(json_path.read_text())) == {
        "segments", "joint_metrics", "skill_summary", "joint_summary", "metadata"
    }
    assert result.metadata["behavioral_thresholds"]["velocity_bands"] == [0.5, 1.0]


def test_analyze_grouped_detector_output_uses_filtered_labels_and_taxonomy(tmp_path):
    input_path = tmp_path / "predicted.h5"
    with h5py.File(input_path, "w") as output:
        data = output.create_group("data")
        for episode_index in range(2):
            episode = data.create_group(f"demo_{episode_index:06d}")
            features = episode.create_dataset(
                "features",
                data=np.array([
                    [0.0, 0.0],
                    [0.5, 0.5],
                    [1.0, 0.5],
                ]),
            )
            features.attrs["feature_names"] = ["joint_pos_1", "joint_vel_1"]
            episode.create_dataset("timestamps/sim", data=[0.0, 0.5, 1.0])
            labels = episode.create_group("labels")
            labels.create_dataset("predicted_skill_id", data=[1, 2, 1])
            filtered = labels.create_dataset("filtered_skill_id", data=[1, 1, 1])
            filtered.attrs["class_skill_ids"] = [1, 2]
            filtered.attrs["class_names_json"] = '["move", "pick"]'

    result = BehavioralAnalyzer().analyze_h5(input_path)

    assert result.segments[["episode_key", "skill_id", "skill"]].to_dict(orient="records") == [
        {"episode_key": "demo_000000", "skill_id": 1, "skill": "move"},
        {"episode_key": "demo_000001", "skill_id": 1, "skill": "move"},
    ]
    assert result.skill_summary.iloc[0]["n_episodes"] == 2
