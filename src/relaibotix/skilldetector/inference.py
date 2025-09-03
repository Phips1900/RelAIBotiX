import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
import hydra


# ----------------- utils -----------------
def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        logging.info(f"Device '{device_str}' unavailable; using 'cpu'.")
        return torch.device("cpu")


def _strip_prefix_in_state_dict(sd: Dict[str, torch.Tensor], prefixes=("model.", "net.", "module.")) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved by Lightning/DataParallel that prefix param names."""
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


# ----------------- models -----------------
def load_model(model_type: str, checkpoint_path: str, window_size: int, num_features: int, num_classes: int):
    if model_type == "cnn_transformer":
        model = CNNTransformer(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    elif model_type == "cnn_lstm":
        model = CNNLSTM(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    elif model_type == "lstm_transformer":
        model = LSTMTransformer(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = _strip_prefix_in_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logging.warning(f"load_state_dict: missing={missing}, unexpected={unexpected}")
    logging.info(f"Loaded {model_type} from {checkpoint_path}")
    return model


# ----------------- data -----------------
def _load_h5_features(h5_path: str, feature_columns: Optional[List[str]] = None,
                      feature_ds: str = "features") -> Tuple[np.ndarray, List[str]]:
    """
    Returns: (features[N,D], feature_names[D])
    If feature_columns is given, will select & reorder columns by matching names found in H5 attrs 'feature_names'.
    """
    with h5py.File(h5_path, "r") as f:
        feats = f[feature_ds][()]  # (N, D_all)
        names_attr = f[feature_ds].attrs.get("feature_names", [])
        feat_names = [(n.decode() if hasattr(n, "decode") else str(n)) for n in names_attr]

    if feature_columns:
        # map requested names to indices in the file
        name_to_idx = {n: i for i, n in enumerate(feat_names)}
        missing = [c for c in feature_columns if c not in name_to_idx]
        if missing:
            raise ValueError(f"Requested feature columns not found in H5: {missing}")
        idxs = [name_to_idx[c] for c in feature_columns]
        feats = feats[:, idxs]
        feat_names = [feat_names[i] for i in idxs]

    feats = feats.astype(np.float32, copy=False)
    return feats, feat_names


class SlidingWindowDataset(Dataset):
    """
    Produces overlapping windows of shape [W, D] with stride S (default 1).
    If N is total timesteps, len(dataset) = N - W + 1 when stride=1.
    """
    def __init__(self, features: np.ndarray, window_size: int, stride: int = 1):
        self.X = features  # (N, D)
        self.W = int(window_size)
        self.S = int(stride)
        self.N = features.shape[0]
        if self.N < self.W:
            raise ValueError(f"Not enough samples N={self.N} for window_size={self.W}")
        self.n_windows = 1 + (self.N - self.W) // self.S

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        i0 = idx * self.S
        i1 = i0 + self.W
        win = self.X[i0:i1]  # (W, D)
        return torch.from_numpy(win)


# ----------------- voting -----------------
def aggregate_predictions(sliding_preds: np.ndarray) -> np.ndarray:
    """
    Majority-vote aggregation from sliding window predictions (n_windows, window_size)
    -> (N,) labels where N = n_windows + window_size - 1 (stride=1).
    """
    n_windows, W = sliding_preds.shape
    N = n_windows + W - 1
    final = np.empty(N, dtype=sliding_preds.dtype)
    for j in range(N):
        i_start = max(0, j - W + 1)
        i_end = min(j, n_windows - 1)
        votes = [sliding_preds[i, j - i] for i in range(i_start, i_end + 1)]
        final[j] = np.argmax(np.bincount(votes))
    return final


# ----------------- writing labels -----------------
def write_labels_to_h5(h5_path: str, labels: np.ndarray, *, overwrite_labels: bool = False,
                       dataset_name_if_exists: str = "labels_pred") -> str:
    """
    Writes labels as int64 to H5 file. If 'labels' exists and overwrite_labels=False,
    creates/overwrites 'labels_pred' instead.
    Returns the dataset name used.
    """
    labels = labels.astype(np.int64, copy=False)
    with h5py.File(h5_path, "a") as f:
        target_name = "labels"
        if "labels" in f and not overwrite_labels:
            target_name = dataset_name_if_exists
            if target_name in f:
                del f[target_name]
        elif "labels" in f and overwrite_labels:
            del f["labels"]
        dset = f.create_dataset(target_name, data=labels, compression="gzip")
    logging.info(f"Wrote labels to '{h5_path}:{target_name}' (len={len(labels)})")
    return target_name


# ----------------- Hydra entry -----------------
@hydra.main(config_path="../configs", config_name="inference_config", version_base=None)
def infer(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("skill_detector")

    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    # 1) Load features (subset to the configured feature names order)
    feats, used_feat_names = _load_h5_features(cfg.data_path, feature_columns=cfg.feature_columns,
                                               feature_ds=getattr(cfg, "feature_dataset", "features"))
    N, D = feats.shape
    logger.info(f"Loaded features: N={N}, D={D}")

    # 2) Load model (select by robot/type from Hydra cfg)
    ckpt_path, model_type = model_select_by_type(cfg.robot, cfg.model_selector)
    model = load_model(model_type, ckpt_path, cfg.window_size, num_features=D, num_classes=cfg.num_classes)
    model.eval().to(device)

    # 3) Build sliding-window dataset & loader
    stride = int(getattr(cfg, "stride", 1))
    ds = SlidingWindowDataset(feats, window_size=cfg.window_size, stride=stride)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=getattr(cfg, "num_workers", 0))
    logger.info(f"Windows: {len(ds)} (W={cfg.window_size}, stride={stride})")

    # 4) Inference over windows
    all_window_preds: List[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            batch = batch.float().to(device)        # [B, W, D]
            logits = model(batch)                    # your models should return [B, W, C]
            if logits.ndim == 2:
                # some models might output [B, C] per window; then repeat per-timestep or adjust your model
                raise RuntimeError("Expected model to output [B, W, C]; got [B, C]. Please adapt.")
            preds = torch.argmax(logits, dim=-1)     # [B, W]
            all_window_preds.append(preds.cpu().numpy())
    dt = time.time() - t0
    logger.info(f"Inference took {dt:.3f}s, {dt/max(len(dl),1):.5f}s/batch")

    if not all_window_preds:
        raise RuntimeError("No predictions produced.")
    sliding_preds = np.concatenate(all_window_preds, axis=0)  # (n_windows, W)

    # 5) Majority voting -> per-timestep labels
    flat_predictions = aggregate_predictions(sliding_preds)   # (N,) when stride=1

    # 6) Write back to H5
    used_ds = write_labels_to_h5(cfg.data_path, flat_predictions,
                                 overwrite_labels=getattr(cfg, "overwrite_labels", False),
                                 dataset_name_if_exists=getattr(cfg, "alt_label_name", "labels_pred"))
    logger.info(f"Done. Labels stored in dataset '{used_ds}'.")
