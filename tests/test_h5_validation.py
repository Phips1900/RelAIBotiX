import h5py
import numpy as np

from relaibotix.data import inspect_h5, validate_h5


def _feature_names(dataset, names):
    dataset.attrs["feature_names"] = np.asarray(names, dtype=h5py.string_dtype("utf-8"))


def test_valid_flat_layout(tmp_path):
    path = tmp_path / "flat.h5"
    with h5py.File(path, "w") as h5_file:
        features = h5_file.create_dataset("features", data=np.ones((6, 2)))
        _feature_names(features, ["joint_pos_1", "joint_vel_1"])
        h5_file.create_dataset("timestamps", data=np.arange(6, dtype=float))
        h5_file.create_dataset("labels", data=np.asarray([0, 0, 0, 1, 1, 1]))

    report = validate_h5(path)
    summary = inspect_h5(path)

    assert report.valid
    assert [warning.code for warning in report.warnings] == ["skills.not_run"]
    assert summary.layout == "flat"
    assert summary.samples == 6
    assert summary.episodes == 2
    assert summary.features == 2


def test_flat_layout_rejects_mismatched_labels(tmp_path):
    path = tmp_path / "invalid.h5"
    with h5py.File(path, "w") as h5_file:
        features = h5_file.create_dataset("features", data=np.ones((6, 1)))
        _feature_names(features, ["joint_pos_1"])
        h5_file.create_dataset("timestamps", data=np.arange(6, dtype=float))
        h5_file.create_dataset("labels", data=np.asarray([0, 0]))

    report = validate_h5(path)

    assert not report.valid
    assert "labels.shape" in {error.code for error in report.errors}


def test_valid_unlabeled_multi_episode_layout(tmp_path):
    path = tmp_path / "mobile.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["schema_version"] = "0.4.0"
        data = h5_file.create_group("data")
        for episode_index in range(2):
            demo = data.create_group(f"demo_{episode_index:06d}")
            features = demo.create_dataset("features", data=np.ones((4, 2)))
            _feature_names(features, ["base_x_m", "base_y_m"])
            demo.create_dataset("timestamps/sim", data=np.arange(4, dtype=float))
            demo.create_dataset("episode/index", data=np.full(4, episode_index))
            demo.create_dataset("labels/skill_id", data=np.full(4, -1))

    report = validate_h5(path)

    assert report.valid
    assert report.summary is not None
    assert report.summary.layout == "multi_episode"
    assert report.summary.episodes == 2
    assert report.summary.samples == 8
    assert [warning.code for warning in report.warnings] == ["skills.not_run"]
