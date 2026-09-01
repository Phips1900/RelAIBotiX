"""Inference-only skill detection for canonical RelAIBotiX HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from relaibotix.data.h5 import decode_feature_names


CANONICAL_SKILL_DATASET = "skills/predicted"


@dataclass(frozen=True)
class InferenceResult:
    h5_path: Path
    dataset: str
    samples: int
    episodes: int
    checkpoint_sha256: str


class SlidingWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, window_size: int):
        self.features = features.astype(np.float32, copy=False)
        self.window_size = int(window_size)
        self.window_count = max(0, len(features) - self.window_size + 1)

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.features[index : index + self.window_size])


class _LearnablePositionEncoding(nn.Module):
    def __init__(self, dimensions: int, dropout: float, window_size: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.empty(1, window_size, dimensions))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1)])


class _FixedPositionEncoding(nn.Module):
    def __init__(self, dimensions: int, dropout: float, window_size: int):
        super().__init__()
        positions = torch.arange(window_size, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimensions, 2, dtype=torch.float) * (-math.log(10_000.0) / dimensions)
        )
        encoding = torch.zeros(window_size, dimensions)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1)])


class _CNNTransformer(nn.Module):
    """Inference architecture matching the training repository's CNN-Transformer."""

    def __init__(
        self,
        *,
        window_size: int,
        num_features: int,
        num_classes: int,
        cnn_channels: Sequence[int],
        transformer: Mapping[str, object],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = num_features
        for output_channels in cnn_channels:
            layers.extend((nn.Conv1d(input_channels, int(output_channels), 3, padding=1), nn.ReLU()))
            input_channels = int(output_channels)
        self.cnn_layers = nn.Sequential(*layers)

        dimensions = int(transformer.get("d_model", cnn_channels[-1]))
        dropout = float(transformer.get("dropout", 0.1))
        encoding = str(transformer.get("encoding", "learnable"))
        if encoding == "learnable":
            self.pos_encoder = _LearnablePositionEncoding(dimensions, dropout, window_size)
        elif encoding == "fixed":
            self.pos_encoder = _FixedPositionEncoding(dimensions, dropout, window_size)
        else:
            raise ValueError(f"Unsupported position encoding: {encoding}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimensions,
            nhead=int(transformer.get("nhead", 8)),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(transformer.get("num_layers", 1)),
        )
        self.fc = nn.Linear(dimensions, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.cnn_layers(values.permute(0, 2, 1)).permute(0, 2, 1)
        return self.fc(self.transformer_encoder(self.pos_encoder(values)))


def _plain_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    return {str(key): item for key, item in dict(value).items()}


def load_model(
    model_type: str,
    checkpoint_path: str | Path,
    *,
    num_features: int,
    window_size: int | None = None,
    num_classes: int | None = None,
) -> tuple[nn.Module, int, int]:
    """Load an inference model and obtain missing dimensions from its checkpoint."""

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    hyperparameters = _plain_mapping(checkpoint.get("hyper_parameters", {}))
    resolved_window = int(window_size or hyperparameters.get("window_size", 0))
    resolved_classes = int(num_classes or hyperparameters.get("num_classes", 0))
    expected_features = int(hyperparameters.get("num_features", num_features))
    if resolved_window <= 0 or resolved_classes <= 0:
        raise ValueError("Checkpoint does not define window_size/num_classes; provide them explicitly.")
    if expected_features != num_features:
        raise ValueError(
            f"Checkpoint expects {expected_features} features, but {num_features} were selected."
        )
    if model_type != "cnn_transformer":
        raise ValueError(f"Unsupported inference model type: {model_type}")

    model = _CNNTransformer(
        window_size=resolved_window,
        num_features=num_features,
        num_classes=resolved_classes,
        cnn_channels=list(hyperparameters.get("cnn_channels", (64, 128))),
        transformer=_plain_mapping(hyperparameters.get("transformer", {})),
    ).float()
    model.load_state_dict(state, strict=True)
    return model, resolved_window, resolved_classes


def aggregate_predictions(sliding_predictions: np.ndarray, num_classes: int) -> np.ndarray:
    """Majority-vote overlapping window predictions into one label per sample."""

    if sliding_predictions.ndim != 2 or sliding_predictions.size == 0:
        raise ValueError("Sliding predictions must have shape [windows, window_size].")
    window_count, window_size = sliding_predictions.shape
    sample_count = window_count + window_size - 1
    votes = np.zeros((sample_count, num_classes), dtype=np.uint32)
    window_indices = np.arange(window_count)
    for offset in range(window_size):
        np.add.at(votes, (window_indices + offset, sliding_predictions[:, offset]), 1)
    return np.argmax(votes, axis=1).astype(np.int64)


def filter_short_segments(labels: np.ndarray, min_length: int) -> np.ndarray:
    """Merge a short label run only when its immediate neighbors agree."""

    output = np.asarray(labels, dtype=np.int64).copy()
    if min_length <= 1 or output.size == 0:
        return output
    starts = np.flatnonzero(np.r_[True, output[1:] != output[:-1]])
    ends = np.r_[starts[1:], output.size]
    original = output.copy()
    for start, end in zip(starts, ends):
        if end - start >= min_length or start == 0 or end == output.size:
            continue
        if original[start - 1] == original[end]:
            output[start:end] = original[start - 1]
    return output


def _device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of: auto, cpu, cuda, mps.")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _episode_bounds(episode_ids: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, episode_ids.size]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _predict_episode(
    model: nn.Module,
    features: np.ndarray,
    *,
    window_size: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    short_episode_policy: str,
) -> np.ndarray:
    original_length = len(features)
    if original_length < window_size:
        if short_episode_policy == "error":
            raise ValueError(
                f"Episode has {original_length} samples; model requires at least {window_size}."
            )
        if short_episode_policy != "pad":
            raise ValueError("Short-episode policy must be 'pad' or 'error'.")
        features = np.pad(
            features,
            ((0, window_size - original_length), (0, 0)),
            mode="edge",
        )
    dataset = SlidingWindowDataset(features, window_size)
    predictions: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch.to(device))
            if logits.shape != (len(batch), window_size, num_classes):
                raise RuntimeError(
                    f"Model returned {tuple(logits.shape)}; expected "
                    f"({len(batch)}, {window_size}, {num_classes})."
                )
            predictions.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return aggregate_predictions(np.concatenate(predictions), num_classes)[:original_length]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_predictions(
    h5_path: Path,
    labels: np.ndarray,
    *,
    dataset_name: str,
    overwrite: bool,
    attributes: Mapping[str, object],
) -> None:
    with h5py.File(h5_path, "a") as output:
        if dataset_name in output and not overwrite:
            raise FileExistsError(
                f"Skill predictions already exist at '/{dataset_name}'. Use overwrite=True to replace them."
            )
        parent_name, _, leaf_name = dataset_name.rpartition("/")
        parent = output.require_group(parent_name) if parent_name else output
        temporary_name = f".{leaf_name}.tmp"
        if temporary_name in parent:
            del parent[temporary_name]
        temporary = parent.create_dataset(
            temporary_name,
            data=labels.astype(np.int64),
            chunks=True,
            compression="gzip",
        )
        for name, value in attributes.items():
            temporary.attrs[name] = value
        if leaf_name in parent:
            del parent[leaf_name]
        parent.move(temporary_name, leaf_name)


def run_inference(
    *,
    h5_path: str | Path,
    checkpoint_path: str | Path,
    model_type: str = "cnn_transformer",
    feature_names: Sequence[str] | None = None,
    feature_columns: Sequence[int] | None = None,
    window_size: int | None = None,
    num_classes: int | None = None,
    batch_size: int = 64,
    device: str = "auto",
    feature_ds: str = "features",
    episode_ids_ds: str = "episode_ids",
    out_labels_name: str = CANONICAL_SKILL_DATASET,
    min_segment_length: int = 10,
    short_episode_policy: str = "pad",
    overwrite: bool = False,
    stride: int = 1,
) -> InferenceResult:
    """Run episode-safe inference and write canonical skill predictions."""

    if stride != 1:
        raise ValueError("Only stride=1 is supported because every sample must receive a prediction.")
    if short_episode_policy not in {"pad", "error"}:
        raise ValueError("Short-episode policy must be 'pad' or 'error'.")
    if (feature_names is None) == (feature_columns is None):
        raise ValueError("Provide exactly one of feature_names or feature_columns.")
    input_path = Path(h5_path)
    checkpoint = Path(checkpoint_path)
    with h5py.File(input_path, "r") as source:
        if out_labels_name in source and not overwrite:
            raise FileExistsError(
                f"Skill predictions already exist at '/{out_labels_name}'. Use overwrite=True to replace them."
            )
        if feature_ds not in source or episode_ids_ds not in source:
            raise ValueError("Inference requires canonical 'features' and 'episode_ids' datasets.")
        feature_dataset = source[feature_ds]
        available_names = decode_feature_names(feature_dataset)
        if feature_names is not None:
            missing = [name for name in feature_names if name not in available_names]
            if missing:
                raise ValueError(f"Input is missing model features: {', '.join(missing)}")
            columns = [available_names.index(name) for name in feature_names]
            selected_names = list(feature_names)
        else:
            columns = [int(column) for column in feature_columns or ()]
            invalid = [column for column in columns if column < 0 or column >= feature_dataset.shape[1]]
            if invalid:
                raise ValueError(f"Feature columns are out of range: {invalid}")
            selected_names = [available_names[column] for column in columns]
        features = feature_dataset[:, columns].astype(np.float32, copy=False)
        episode_ids = source[episode_ids_ds][:].reshape(-1)
    if features.shape[0] != episode_ids.size:
        raise ValueError("Features and episode IDs must have equal lengths.")
    if features.shape[0] == 0:
        raise ValueError("Skill inference requires at least one sample.")

    model, resolved_window, resolved_classes = load_model(
        model_type,
        checkpoint,
        num_features=features.shape[1],
        window_size=window_size,
        num_classes=num_classes,
    )
    selected_device = _device(device)
    model = model.to(selected_device).eval()
    predictions = np.empty(features.shape[0], dtype=np.int64)
    bounds = _episode_bounds(episode_ids)
    for start, end in bounds:
        episode_predictions = _predict_episode(
            model,
            features[start:end],
            window_size=resolved_window,
            num_classes=resolved_classes,
            batch_size=batch_size,
            device=selected_device,
            short_episode_policy=short_episode_policy,
        )
        predictions[start:end] = filter_short_segments(episode_predictions, min_segment_length)

    checkpoint_hash = _sha256(checkpoint)
    _write_predictions(
        input_path,
        predictions,
        dataset_name=out_labels_name,
        overwrite=overwrite,
        attributes={
            "model_type": model_type,
            "checkpoint_file": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            "feature_names_json": json.dumps(selected_names),
            "window_size": resolved_window,
            "num_classes": resolved_classes,
            "min_segment_length": int(min_segment_length),
            "short_episode_policy": short_episode_policy,
            "episode_safe": True,
        },
    )
    return InferenceResult(input_path, out_labels_name, len(predictions), len(bounds), checkpoint_hash)
