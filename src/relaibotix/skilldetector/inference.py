"""Adapter for the separately maintained RelAIBotiX skill detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py


@dataclass(frozen=True)
class InferenceResult:
    input_h5: Path
    output_h5: Path
    checkpoint: Path
    samples: int
    episodes: int
    modality: str


def _prediction_summary(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as source:
        data = source.get("data")
        if not isinstance(data, h5py.Group):
            raise ValueError("Skill-detector output must contain the canonical '/data' episode group.")
        episodes = 0
        samples = 0
        for key in sorted(data):
            episode = data[key]
            if not isinstance(episode, h5py.Group) or "features" not in episode:
                continue
            if "labels/filtered_skill_id" not in episode and "labels/predicted_skill_id" not in episode:
                raise ValueError(f"Detector output is missing predictions for /data/{key}.")
            episodes += 1
            samples += int(episode["features"].shape[0])
        if not episodes:
            raise ValueError("Skill-detector output contains no episodes.")
        return samples, episodes


def run_inference(
    *,
    h5_path: str | Path,
    checkpoint_path: str | Path,
    output_h5: str | Path,
    modality: str = "timeseries",
    lerobot_root: str | Path | None = None,
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "auto",
    minimum_skill_frames: int = 5,
    target_stride: int = 1,
) -> InferenceResult:
    """Run a detector checkpoint and write predictions to a separate HDF5 copy.

    Feature selection, normalization, architecture, taxonomy, and window alignment
    are read from the checkpoint by the external detector package.
    """

    input_path = Path(h5_path)
    output_path = Path(output_h5)
    checkpoint = Path(checkpoint_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output HDF5 must differ from input; source data is never modified.")

    try:
        if modality == "timeseries":
            from relaibotix_skill_detector.timeseries import predict_timeseries

            written = predict_timeseries(
                input_path,
                checkpoint,
                output_path,
                batch_size,
                num_workers,
                device,
                minimum_skill_frames,
            )
        elif modality == "camera":
            if lerobot_root is None:
                raise ValueError("Camera inference requires --lerobot-root.")
            from relaibotix_skill_detector.camera import predict_camera

            written = predict_camera(
                input_path,
                Path(lerobot_root),
                checkpoint,
                output_path,
                batch_size,
                target_stride,
                minimum_skill_frames,
                num_workers,
                device,
            )
        elif modality == "hybrid":
            if lerobot_root is None:
                raise ValueError("Hybrid inference requires --lerobot-root.")
            from relaibotix_skill_detector.hybrid import predict_hybrid

            written = predict_hybrid(
                input_path,
                Path(lerobot_root),
                checkpoint,
                output_path,
                batch_size,
                target_stride,
                minimum_skill_frames,
                num_workers,
                device,
            )
        else:
            raise ValueError("Modality must be one of: timeseries, camera, hybrid.")
    except ImportError as error:
        raise RuntimeError(
            "Skill inference requires the 'skill-detection' extra: "
            "pip install 'relaibotix[skill-detection]'"
        ) from error

    written_path = Path(written)
    samples, episodes = _prediction_summary(written_path)
    return InferenceResult(input_path, written_path, checkpoint, samples, episodes, modality)
