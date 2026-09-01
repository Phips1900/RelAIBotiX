import h5py
import numpy as np
import pytest
import torch

from relaibotix.behavioral import BehavioralAnalyzer
from relaibotix.skilldetector import inference


class FakeSkillModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_batches = []

    def forward(self, features):
        self.seen_batches.append(features.detach().cpu().numpy())
        labels = (features[:, :, 0] > 0.5).long()
        return torch.nn.functional.one_hot(labels, num_classes=2).float()


def _write_canonical(path):
    with h5py.File(path, "w") as output:
        features = output.create_dataset(
            "features",
            data=np.array([
                [0.0, 0.0],
                [0.0, 0.1],
                [0.0, 0.2],
                [1.0, 0.3],
                [1.0, 0.4],
                [1.0, 0.5],
            ]),
        )
        features.attrs["feature_names"] = ["joint_pos_1", "joint_vel_1"]
        output.create_dataset("timestamps", data=[0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        output.create_dataset("episode_ids", data=[0, 0, 0, 1, 1, 1])


def test_aggregate_predictions_majority_votes_overlap():
    windows = np.array([
        [0, 1, 1],
        [1, 1, 0],
    ])

    assert inference.aggregate_predictions(windows, num_classes=2).tolist() == [0, 1, 1, 0]


def test_inference_writes_canonical_labels_without_crossing_episodes(tmp_path, monkeypatch):
    input_path = tmp_path / "input.h5"
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    _write_canonical(input_path)
    model = FakeSkillModel()
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda *args, **kwargs: (model, 2, 2),
    )

    result = inference.run_inference(
        h5_path=input_path,
        checkpoint_path=checkpoint,
        feature_names=["joint_pos_1", "joint_vel_1"],
        device="cpu",
        min_segment_length=0,
    )

    assert result.dataset == "skills/predicted"
    assert result.episodes == 2
    assert all(np.unique(batch[:, :, 0]).size == 1 for batch in model.seen_batches)
    with h5py.File(input_path, "r") as source:
        predicted = source["skills/predicted"]
        assert predicted[:].tolist() == [0, 0, 0, 1, 1, 1]
        assert predicted.attrs["episode_safe"]
        assert predicted.attrs["feature_names_json"] == '["joint_pos_1", "joint_vel_1"]'

    behavior = BehavioralAnalyzer().analyze_h5(input_path)
    assert len(behavior.segments) == 2

    with pytest.raises(FileExistsError, match="already exist"):
        inference.run_inference(
            h5_path=input_path,
            checkpoint_path=checkpoint,
            feature_names=["joint_pos_1", "joint_vel_1"],
            device="cpu",
        )


def test_inference_can_reject_episode_shorter_than_window(tmp_path, monkeypatch):
    input_path = tmp_path / "input.h5"
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    _write_canonical(input_path)
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda *args, **kwargs: (FakeSkillModel(), 4, 2),
    )

    with pytest.raises(ValueError, match="requires at least 4"):
        inference.run_inference(
            h5_path=input_path,
            checkpoint_path=checkpoint,
            feature_names=["joint_pos_1", "joint_vel_1"],
            device="cpu",
            short_episode_policy="error",
        )


def test_inference_pads_short_episodes_without_crossing_boundaries(tmp_path, monkeypatch):
    input_path = tmp_path / "input.h5"
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    _write_canonical(input_path)
    model = FakeSkillModel()
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda *args, **kwargs: (model, 4, 2),
    )

    inference.run_inference(
        h5_path=input_path,
        checkpoint_path=checkpoint,
        feature_names=["joint_pos_1", "joint_vel_1"],
        device="cpu",
        min_segment_length=0,
    )

    with h5py.File(input_path, "r") as source:
        assert source["skills/predicted"][:].tolist() == [0, 0, 0, 1, 1, 1]
        assert source["skills/predicted"].attrs["short_episode_policy"] == "pad"
    assert all(np.unique(batch[:, :, 0]).size == 1 for batch in model.seen_batches)
