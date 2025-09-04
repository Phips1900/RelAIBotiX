from pathlib import Path
from typing import Sequence, Union, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def aggregate_predictions(sliding_preds: np.ndarray) -> np.ndarray:
    n_windows, W = sliding_preds.shape
    N = n_windows + W - 1
    out = np.empty(N, dtype=sliding_preds.dtype)
    for j in range(N):
        i0 = max(0, j - W + 1)
        i1 = min(j, n_windows - 1)
        votes = [sliding_preds[i, j - i] for i in range(i0, i1 + 1)]
        out[j] = np.argmax(np.bincount(votes))
    return out


# ---- minimal dataset for sliding-window inference ----
class SlidingWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, window_size: int, stride: int = 1):
        self.X = X.astype(np.float32, copy=False)
        self.W = int(window_size)
        self.stride = int(stride)
        self.n = max(0, (len(X) - self.W) // self.stride + 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        i = idx * self.stride
        return torch.from_numpy(self.X[i:i + self.W, :])  # [W, D]


# ---- model loader (no inference_engine) ----
def load_model(model_type: str, checkpoint_path: Union[str, Path],
               window_size: int, num_features: int, num_classes: int) -> torch.nn.Module:
    # import your model classes directly
    if model_type == "cnn_transformer":
        from .models.cnn_transformer import CNNTransformer as Model
        model = Model(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    elif model_type == "cnn_lstm":
        from .models.cnn_lstm import CNNLSTM as Model
        model = Model(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    elif model_type == "lstm_transformer":
        from .models.lstm_transformer import LSTMTransformer as Model
        model = Model(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)  # handle lightning or plain
    model.load_state_dict(state, strict=False)
    return model


def _write_labels_to_h5(h5_path: Union[str, Path], labels: np.ndarray,
                        *, dataset_name: str = "labels_pred", overwrite: bool = True) -> str:
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "a") as f:
        if dataset_name in f and overwrite:
            del f[dataset_name]
        if dataset_name not in f:
            f.create_dataset(dataset_name, data=labels.astype(np.int64))
    return dataset_name


def run_inference(
        *,
        h5_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        model_type: str,
        window_size: int,
        feature_columns: Sequence[int],
        num_classes: int,
        batch_size: int = 64,
        device: str = "cpu",  # "cuda"/"mps"/"cpu"
        feature_ds: str = "features",
        out_labels_name: str = "labels_pred",
        stride: int = 1,
) -> str:
    """Reads features from H5, runs model, majority-votes, writes `labels_pred`. Returns dataset name."""
    h5_path = Path(h5_path)

    # 1) features by index (simple & robust)
    with h5py.File(h5_path, "r") as f:
        X = f[feature_ds][()]  # (N, D)
    cols = [int(c) for c in feature_columns]
    D = X.shape[1]
    bad = [i for i in cols if i < 0 or i >= D]
    if bad:
        raise ValueError(f"indices out of range for D={D}: {bad}")
    feats = X[:, cols].astype(np.float32, copy=False)  # (N, len(cols))

    # 2) model
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available()
                       else "mps" if device == "mps" and torch.backends.mps.is_available() else "cpu")
    model = load_model(model_type, checkpoint_path, window_size, feats.shape[1], num_classes).to(dev).eval()

    # 3) sliding-window inference
    ds = SlidingWindowDataset(feats, window_size=window_size, stride=stride)
    if len(ds) == 0:
        raise ValueError("Not enough samples for the chosen window_size.")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds_windows = []
    with torch.no_grad():
        for batch in dl:
            x = batch.to(dev)  # [B, W, D]
            logits = model(x)  # expect [B, W, C]
            if logits.dim() != 3:
                raise RuntimeError(f"Unsupported model output shape: {tuple(logits.shape)}; expected [B, W, C].")
            preds = torch.argmax(logits, dim=-1)  # [B, W]
            preds_windows.append(preds.cpu().numpy())
    sliding_preds = np.concatenate(preds_windows, axis=0)  # [n_windows, W]

    # 4) majority vote to per-timestep labels
    flat_preds = aggregate_predictions(sliding_preds)  # [N]

    # 5) write labels_pred
    return _write_labels_to_h5(h5_path, flat_preds, dataset_name=out_labels_name, overwrite=True)
