import h5py
import numpy as np

from relaibotix.data import convert_h5, inspect_h5, validate_h5


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
    assert "episode_ids.shape" in {error.code for error in report.errors}


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


def test_convert_flat_preserves_old_predictions_as_source_data(tmp_path):
    source = tmp_path / "legacy.h5"
    output = tmp_path / "canonical.h5"
    with h5py.File(source, "w") as h5_file:
        features = h5_file.create_dataset("features", data=np.arange(12).reshape(6, 2))
        _feature_names(features, ["joint_pos_1", "joint_vel_1"])
        h5_file.create_dataset("timestamps", data=np.arange(6, dtype=float))
        h5_file.create_dataset("labels", data=np.asarray([0, 0, 0, 1, 1, 1]))
        h5_file.create_dataset("predicted_labels", data=np.asarray([0, 0, 1, 1, 1, 0]))

    converted = convert_h5(source, output)
    report = validate_h5(converted)

    assert report.valid
    assert [warning.code for warning in report.warnings] == ["skills.not_run"]
    with h5py.File(output, "r") as h5_file:
        assert h5_file.attrs["format"] == "relaibotix_hdf5"
        assert h5_file.attrs["schema_version"] == "1.0"
        np.testing.assert_array_equal(h5_file["episode_ids"][:], [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(
            h5_file["source/predicted_labels"][:],
            [0, 0, 1, 1, 1, 0],
        )
        assert "predicted_labels" not in h5_file


def test_convert_multi_episode_flattens_episodes(tmp_path):
    source = tmp_path / "mobile.h5"
    output = tmp_path / "canonical.h5"
    with h5py.File(source, "w") as h5_file:
        data = h5_file.create_group("data")
        for episode_index, length in enumerate((3, 2)):
            demo = data.create_group(f"demo_{episode_index:06d}")
            features = demo.create_dataset(
                "features",
                data=np.full((length, 2), episode_index, dtype=float),
            )
            _feature_names(features, ["base_x_m", "base_y_m"])
            demo.create_dataset("timestamps/sim", data=np.arange(length, dtype=float))
            demo.create_dataset("episode/index", data=np.full(length, episode_index))
            demo.create_dataset("labels/skill_id", data=np.full(length, -1))

    convert_h5(source, output)
    summary = inspect_h5(output)

    assert summary.layout == "flat"
    assert summary.episodes == 2
    assert summary.samples == 5
    with h5py.File(output, "r") as h5_file:
        np.testing.assert_array_equal(h5_file["episode_ids"][:], [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(h5_file["source/skill_ids"][:], [-1, -1, -1, -1, -1])


def test_convert_renumbers_repeated_flat_episode_ids(tmp_path):
    source = tmp_path / "repeated.h5"
    output = tmp_path / "canonical.h5"
    with h5py.File(source, "w") as h5_file:
        features = h5_file.create_dataset("features", data=np.ones((6, 1)))
        _feature_names(features, ["joint_pos_1"])
        h5_file.create_dataset("timestamps", data=np.arange(6, dtype=float))
        h5_file.create_dataset("labels", data=[0, 0, 1, 1, 0, 0])

    convert_h5(source, output)

    with h5py.File(output, "r") as converted:
        assert converted["episode_ids"][:].tolist() == [0, 0, 1, 1, 2, 2]
    assert inspect_h5(output).episodes == 3
