import shutil
import sys
from types import ModuleType

import h5py
import numpy as np
import pytest

from relaibotix.skilldetector.inference import run_inference


def _write_grouped_input(path):
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        for episode_index in range(2):
            episode = data.create_group(f"demo_{episode_index:06d}")
            features = episode.create_dataset(
                "features",
                data=np.ones((3, 2), dtype=float),
            )
            features.attrs["feature_names"] = ["joint_pos_1", "joint_vel_1"]
            episode.create_dataset("timestamps/sim", data=[0.0, 0.5, 1.0])
            episode.create_dataset("labels/skill_id", data=[-1, -1, -1])


def _install_fake_detector(monkeypatch):
    package = ModuleType("relaibotix_skill_detector")
    package.__path__ = []
    timeseries = ModuleType("relaibotix_skill_detector.timeseries")

    def predict_timeseries(
        input_path,
        checkpoint,
        output_path,
        batch_size,
        workers,
        device,
        minimum_frames,
    ):
        shutil.copy2(input_path, output_path)
        with h5py.File(output_path, "r+") as output:
            for episode in output["data"].values():
                labels = episode["labels"]
                predicted = labels.create_dataset("predicted_skill_id", data=[1, 2, 2])
                predicted.attrs["class_skill_ids"] = [1, 2]
                predicted.attrs["class_names_json"] = '["move", "pick"]'
                filtered = labels.create_dataset("filtered_skill_id", data=[1, 1, 2])
                filtered.attrs["class_skill_ids"] = [1, 2]
                filtered.attrs["class_names_json"] = '["move", "pick"]'
        return output_path

    timeseries.predict_timeseries = predict_timeseries
    monkeypatch.setitem(sys.modules, "relaibotix_skill_detector", package)
    monkeypatch.setitem(sys.modules, "relaibotix_skill_detector.timeseries", timeseries)


def test_inference_delegates_and_preserves_source(tmp_path, monkeypatch):
    _install_fake_detector(monkeypatch)
    input_path = tmp_path / "input.h5"
    output_path = tmp_path / "predicted.h5"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    _write_grouped_input(input_path)

    result = run_inference(
        h5_path=input_path,
        checkpoint_path=checkpoint,
        output_h5=output_path,
        device="cpu",
    )

    assert result.samples == 6
    assert result.episodes == 2
    assert result.output_h5 == output_path
    with h5py.File(input_path, "r") as source:
        assert "labels/predicted_skill_id" not in source["data/demo_000000"]
    with h5py.File(output_path, "r") as output:
        assert output["data/demo_000000/labels/filtered_skill_id"][:].tolist() == [1, 1, 2]


def test_inference_refuses_to_modify_source_in_place(tmp_path):
    input_path = tmp_path / "input.h5"
    _write_grouped_input(input_path)

    with pytest.raises(ValueError, match="must differ"):
        run_inference(
            h5_path=input_path,
            checkpoint_path=tmp_path / "model.pt",
            output_h5=input_path,
        )


def test_camera_and_hybrid_require_video_root(tmp_path):
    for modality in ("camera", "hybrid"):
        with pytest.raises(ValueError, match="lerobot-root"):
            run_inference(
                h5_path=tmp_path / "input.h5",
                checkpoint_path=tmp_path / "model.pt",
                output_h5=tmp_path / f"{modality}.h5",
                modality=modality,
            )
