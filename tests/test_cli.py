import h5py
import numpy as np
from types import SimpleNamespace

from relaibotix.cli import main


def _write_valid_input(path):
    with h5py.File(path, "w") as h5_file:
        features = h5_file.create_dataset("features", data=np.ones((4, 2)))
        features.attrs["feature_names"] = np.asarray(
            ["joint_pos_1", "joint_vel_1"],
            dtype=h5py.string_dtype("utf-8"),
        )
        h5_file.create_dataset("timestamps", data=np.arange(4, dtype=float))
        h5_file.create_dataset("labels", data=np.asarray([0, 0, 1, 1]))


def test_h5_validate_command(tmp_path, capsys):
    path = tmp_path / "input.h5"
    _write_valid_input(path)

    exit_code = main(["h5", "validate", str(path)])

    assert exit_code == 0
    assert "VALID:" in capsys.readouterr().out


def test_h5_convert_command(tmp_path, capsys):
    source = tmp_path / "input.h5"
    output = tmp_path / "converted.h5"
    _write_valid_input(source)

    exit_code = main(["h5", "convert", str(source), str(output)])

    assert exit_code == 0
    assert output.exists()
    text = capsys.readouterr().out
    assert "Converted:" in text
    assert "VALID:" in text


def test_behavior_command(tmp_path):
    input_path = tmp_path / "labeled.h5"
    output_path = tmp_path / "behavior"
    with h5py.File(input_path, "w") as output:
        features = output.create_dataset(
            "features",
            data=np.array([
                [0.0, 0.0],
                [1.0, 0.5],
                [0.5, -0.5],
            ]),
        )
        features.attrs["feature_names"] = ["joint_pos_1", "joint_vel_1"]
        output.create_dataset("timestamps", data=[0.0, 0.5, 1.0])
        output.create_dataset("episode_ids", data=[0, 0, 0])
        skills = output.create_group("skills")
        skills.create_dataset("predicted", data=[0, 0, 1])

    assert main(["behavior", str(input_path), "--output", str(output_path)]) == 0
    assert (output_path / "segments.csv").is_file()
    assert (output_path / "joint_metrics.csv").is_file()
    assert (output_path / "behavior.json").is_file()


def test_skills_infer_command(tmp_path, monkeypatch):
    import relaibotix.skilldetector

    captured = {}

    def fake_inference(**arguments):
        captured.update(arguments)
        return SimpleNamespace(samples=12, episodes=2, dataset="skills/predicted")

    monkeypatch.setattr(relaibotix.skilldetector, "run_inference", fake_inference)
    input_path = tmp_path / "input.h5"
    checkpoint = tmp_path / "model.ckpt"

    assert main([
        "skills",
        "infer",
        str(input_path),
        "--checkpoint",
        str(checkpoint),
        "--features",
        "x",
        "gripper_state",
    ]) == 0
    assert captured["feature_names"] == ["x", "gripper_state"]
    assert captured["h5_path"] == input_path
