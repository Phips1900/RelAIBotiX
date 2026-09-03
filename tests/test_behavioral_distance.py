import numpy as np
import pytest

from relaibotix.behavioral import BehavioralAnalyzer


def _analyze_positions(positions):
    sample_count = len(positions)
    return BehavioralAnalyzer().analyze(
        features=np.asarray(positions, dtype=float).reshape(-1, 1),
        feature_names=["joint_pos_1"],
        skill_labels=np.zeros(sample_count, dtype=int),
        timestamps=np.arange(sample_count, dtype=float),
        episode_ids=np.zeros(sample_count, dtype=int),
    )


def test_joint_distance_is_position_path_length():
    result = _analyze_positions([0.0, 1.0, 0.25, 0.5])

    assert result.joint_metrics.iloc[0]["traveled_distance"] == pytest.approx(2.0)


def test_joint_distance_does_not_bridge_missing_samples():
    result = _analyze_positions([0.0, 1.0, np.nan, 4.0, 5.5])

    assert result.joint_metrics.iloc[0]["traveled_distance"] == pytest.approx(2.5)


def test_summary_reports_distance_by_skill_and_joint():
    result = _analyze_positions([0.0, 1.0, 0.25, 0.5])
    distance = result.joint_summary.iloc[0]

    assert distance["skill_id"] == 0
    assert distance["joint"] == "j1"
    assert distance["total_traveled_distance"] == pytest.approx(2.0)
    assert distance["mean_traveled_distance"] == pytest.approx(2.0)
