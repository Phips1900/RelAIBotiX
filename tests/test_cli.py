import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from types import SimpleNamespace

from relaibotix.cli import main


def _write_robot_config(path, *, velocity_bands=(0.5, 1.0)):
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "robot": {"id": "test_robot", "name": "Test Robot", "type": "test"},
        "failure_probability_basis": "per_minute",
        "failure_probability_source": "test_source",
        "exposure_assumptions": {
            "source": "test_expert",
            "position_step": 0.001,
            "velocity_active": 0.03,
            "effort_active": 0.1,
            "velocity_bands": list(velocity_bands),
            "effort_bands": [0.2, 0.6],
            "velocity_multipliers": [1.0, 1.5, 2.0],
            "effort_multipliers": [1.0, 1.5, 2.0],
            "distance_multipliers": [1.0, 1.5, 2.0],
        },
        "components": {
            "joint_1": {
                "type": "revolute_joint",
                "features": {
                    "position": "joint_pos_1",
                    "velocity": "joint_vel_1",
                },
                "failure_probability": 0.01,
                "redundancy": {"copies": 1, "mode": "parallel"},
            },
            "controller": {
                "type": "controller",
                "always_active": True,
                "failure_probability": 0.01,
                "redundancy": {"copies": 1, "mode": "parallel"},
            }
        },
    }))


def _write_valid_input(path):
    with h5py.File(path, "w") as h5_file:
        features = h5_file.create_dataset("features", data=np.ones((4, 2)))
        features.attrs["feature_names"] = np.asarray(
            ["joint_pos_1", "joint_vel_1"],
            dtype=h5py.string_dtype("utf-8"),
        )
        h5_file.create_dataset("timestamps", data=np.arange(4, dtype=float))
        h5_file.create_dataset("labels", data=np.asarray([0, 0, 1, 1]))


def _write_canonical_input(path):
    with h5py.File(path, "w") as h5_file:
        data = h5_file.create_group("data")
        episode = data.create_group("demo_000000")
        features = episode.create_dataset(
            "features",
            data=np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.5]]),
        )
        features.attrs["feature_names"] = ["joint_pos_1", "joint_vel_1"]
        episode.create_dataset("timestamps/sim", data=[0.0, 0.5, 1.0])
        episode.create_dataset("episode/index", data=[0, 0, 0])
        episode.create_dataset("labels/skill_id", data=[-1, -1, -1])


def test_h5_validate_command(tmp_path, capsys):
    path = tmp_path / "input.h5"
    _write_valid_input(path)

    exit_code = main(["h5", "validate", str(path)])

    assert exit_code == 0
    assert "VALID:" in capsys.readouterr().out


def test_h5_validate_checks_robot_config_features(tmp_path, capsys):
    path = tmp_path / "input.h5"
    config = tmp_path / "robot.json"
    _write_valid_input(path)
    _write_robot_config(config)

    assert main(["h5", "validate", str(path), "--config", str(config)]) == 0
    assert "config.features_missing" not in capsys.readouterr().out

    with h5py.File(path, "r+") as h5_file:
        h5_file["features"].attrs["feature_names"] = ["joint_pos_1", "unused"]

    assert main(["h5", "validate", str(path), "--config", str(config)]) == 1
    output = capsys.readouterr().out
    assert "config.features_missing" in output
    assert "joint_vel_1" in output


def test_no_arguments_shows_new_cli_help(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "{h5,config,skills,behavior,reliability,experiments,run,gui}" in output
    assert "--ckpt" not in output


def test_config_validate_command(capsys):
    assert main(["config", "validate", "configs/robots/so_arm.json"]) == 0
    output = capsys.readouterr().out
    assert "VALID:" in output
    assert "SO-ARM 101 (so_arm_101)" in output
    assert "Measured components: 6" in output


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
    config_path = tmp_path / "robot.json"
    _write_robot_config(config_path, velocity_bands=(0.25, 0.75))
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

    assert main([
        "behavior",
        str(input_path),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
    ]) == 0
    assert (output_path / "segments.csv").is_file()
    assert (output_path / "joint_metrics.csv").is_file()
    assert (output_path / "behavior.json").is_file()
    payload = json.loads((output_path / "behavior.json").read_text())
    assert payload["metadata"]["behavioral_thresholds"]["velocity_bands"] == [0.25, 0.75]


def test_skills_infer_command(tmp_path, monkeypatch):
    import relaibotix.skilldetector

    captured = {}

    def fake_inference(**arguments):
        captured.update(arguments)
        return SimpleNamespace(samples=12, episodes=2, output_h5=arguments["output_h5"])

    monkeypatch.setattr(relaibotix.skilldetector, "run_inference", fake_inference)
    input_path = tmp_path / "input.h5"
    checkpoint = tmp_path / "model.ckpt"
    output_path = tmp_path / "labeled.h5"

    assert main([
        "skills",
        "infer",
        str(input_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_path),
    ]) == 0
    assert captured["h5_path"] == input_path
    assert captured["output_h5"] == output_path


def test_skills_list_command(capsys):
    assert main(["skills", "list"]) == 0
    output = capsys.readouterr().out
    assert "mobile-lstm: mobile, timeseries [recommended]" in output
    assert "franka-sim-lstm: franka_sim, timeseries [recommended]" in output


def test_run_command_executes_complete_pipeline(tmp_path, monkeypatch):
    import relaibotix.cli as cli

    input_path = tmp_path / "input.h5"
    config_path = tmp_path / "robot.json"
    output_path = tmp_path / "run"
    checkpoint = tmp_path / "model.pt"
    _write_canonical_input(input_path)
    _write_robot_config(config_path)
    checkpoint.write_bytes(b"checkpoint")

    def fake_skills(arguments):
        source = Path(arguments[1])
        destination = Path(arguments[arguments.index("--output") + 1])
        shutil.copy2(source, destination)
        with h5py.File(destination, "r+") as output:
            labels = output["data/demo_000000/labels"]
            labels.create_dataset("predicted_skill_id", data=[1, 1, 1])
            labels.create_dataset("filtered_skill_id", data=[1, 1, 1])
        return 0

    monkeypatch.setattr(cli, "_run_skills", fake_skills)
    assert main([
        "run",
        str(input_path),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_path),
        "--device",
        "cpu",
        "--sensitivity",
    ]) == 0

    assert (output_path / "predicted.h5").is_file()
    assert (output_path / "behavior" / "behavior.json").is_file()
    assert (output_path / "reliability" / "reliability.json").is_file()
    assert (output_path / "reliability" / "model.pm").is_file()
    assert (output_path / "reliability" / "model_repeated_runs.pm").is_file()
    assert (output_path / "reliability" / "sensitivity.csv").is_file()


def test_run_command_can_reproduce_legacy_stored_predictions(tmp_path):
    input_path = tmp_path / "legacy.h5"
    config_path = tmp_path / "robot.json"
    output_path = tmp_path / "run"
    _write_valid_input(input_path)
    with h5py.File(input_path, "r+") as h5_file:
        h5_file.create_dataset("skills/predicted", data=np.asarray([0, 0, 1, 1]))
    _write_robot_config(config_path)

    assert main([
        "run",
        str(input_path),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--legacy-existing-predictions",
        "--sensitivity",
    ]) == 0

    assert not (output_path / "canonical.h5").exists()
    assert not (output_path / "predicted.h5").exists()
    assert (output_path / "behavior" / "behavior.json").is_file()
    assert (output_path / "reliability" / "reliability.json").is_file()


def test_reliability_command(tmp_path):
    behavior_path = tmp_path / "behavior.json"
    output_path = tmp_path / "reliability"
    behavior_path.write_text(
        '{"segments":[{"episode_key":"demo_0","skill_id":1,"start_index":0}],'
        '"joint_metrics":[],'
        '"skill_summary":[{"skill_id":1,"skill":"move","n_episodes":1,"n_segments":1,'
        '"total_duration":10.0}],"joint_summary":[]}'
    )
    config_path = tmp_path / "robot.json"
    _write_robot_config(config_path)

    assert main([
        "reliability",
        str(behavior_path),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--sensitivity",
    ]) == 0
    assert (output_path / "component_failures.csv").is_file()
    assert (output_path / "skill_probabilities.csv").is_file()
    assert (output_path / "reliability.json").is_file()
    assert (output_path / "model.pm").is_file()
    assert (output_path / "model.pctl").is_file()
    assert (output_path / "model_repeated_runs.pm").is_file()
    assert (output_path / "model_repeated_runs.pctl").is_file()
    assert (output_path / "sensitivity.csv").is_file()
    assert (output_path / "sensitivity.json").is_file()
    reliability = json.loads((output_path / "reliability.json").read_text())
    assert reliability["repeated_run_mttf"]["hours"] > 0.0


def test_experiments_command_writes_publication_outputs(tmp_path):
    input_path = tmp_path / "input.h5"
    config_path = tmp_path / "robot.json"
    manifest_path = tmp_path / "experiments.json"
    output_path = tmp_path / "publication"
    _write_valid_input(input_path)
    with h5py.File(input_path, "r+") as h5_file:
        h5_file.create_dataset("skills/predicted", data=np.asarray([0, 0, 1, 1]))
    _write_robot_config(config_path)
    manifest_path.write_text(json.dumps({
        "schema_version": "1.0",
        "name": "Test experiments",
        "experiments": [{
            "id": "test_policy",
            "setting": "CS-test",
            "platform": "Test Robot",
            "task": "test task",
            "policy": "test policy",
            "input_h5": "input.h5",
            "robot_config": "robot.json",
            "scope": "included",
            "skill_names": {"0": "Move", "1": "Place"},
            "label_source": "existing_detector_predictions",
        }],
    }))

    assert main([
        "experiments",
        "run",
        str(manifest_path),
        "--output",
        str(output_path),
    ]) == 0

    experiment = output_path / "test_policy"
    assert (experiment / "behavior" / "behavior.json").is_file()
    assert (experiment / "reliability" / "reliability.json").is_file()
    assert (experiment / "reliability" / "sensitivity.csv").is_file()
    for filename in (
        "paper_results.csv",
        "paper_results.md",
        "paper_results.tex",
        "provenance.json",
    ):
        assert (output_path / filename).is_file()
    assert "Move" in (experiment / "behavior" / "behavior.json").read_text()
    assert "MTTF (h)" in (output_path / "paper_results.tex").read_text()
    provenance = json.loads((output_path / "provenance.json").read_text())
    assert provenance["solvers"]["internal"]["enabled"] is True
    assert provenance["experiments"][0]["input_sha256"]


def test_gui_command_dispatches_without_importing_qt(monkeypatch):
    import relaibotix.gui

    monkeypatch.setattr(relaibotix.gui, "launch_gui", lambda: 17)
    assert main(["gui"]) == 17
