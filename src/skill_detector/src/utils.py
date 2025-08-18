import h5py
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import logging
import hydra
import warnings
import torch

class RoboDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RoboWindowedDataset(Dataset):
    def __init__(self, X, y, window_size):
        """
        X: np.ndarray or torch.Tensor of shape (N, num_features)
        y: one-hot encoded labels of shape (N, num_classes)
        window_size: number of timesteps per sample
        """
        self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        self.y = torch.tensor(y, dtype=torch.float32) if not torch.is_tensor(y) else y
        self.window_size = window_size

        self.num_windows = len(self.X) - window_size + 1
        if self.num_windows < 1:
            raise ValueError(f"window_size={window_size} is too large for dataset length {len(self.X)}")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        window_x = self.X[idx:idx + self.window_size]
        window_y = self.y[idx:idx + self.window_size]
        return window_x, window_y


class RoboDatasetInference(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

class RoboWindowedDatasetInference(Dataset):
    def __init__(self, X, window_size):
        """
        X: np.ndarray or torch.Tensor of shape (N, num_features)
        y: one-hot encoded labels of shape (N, num_classes)
        window_size: number of timesteps per sample
        """
        self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
        self.window_size = window_size

        self.num_windows = len(self.X) - window_size + 1
        if self.num_windows < 1:
            raise ValueError(f"window_size={window_size} is too large for dataset length {len(self.X)}")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        window_x = self.X[idx:idx + self.window_size]
        return window_x


def setup_logger():
    run_dir = HydraConfig.get().runtime.output_dir
    log_file = f"{run_dir}/training_pipeline.log"

    logger = logging.getLogger()  # Root logger
    logger.handlers = []  # Clear existing handlers

    # Set up handlers (append mode for FileHandler)
    handlers = [
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

def validate_h5_structure_in_train(file_path: str, data_cfg: DictConfig):
    with h5py.File(file_path, 'r') as f:
        if data_cfg.structure.features_key not in f:
            raise KeyError(f"Features key '{data_cfg.structure.features_key}' not found in HDF5 file")
        if data_cfg.structure.labels_key not in f:
            raise KeyError(f"Labels key '{data_cfg.structure.labels_key}' not found in HDF5 file")

        features = f[data_cfg.structure.features_key]
        if max(data_cfg.feature_columns) >= features.shape[1]:
            raise ValueError(f"Feature columns exceed available features (max index: {features.shape[1] - 1})")

def validate_h5_structure_in_inference(file_path: str, inference_cfg: DictConfig):
    with h5py.File(file_path, 'r') as f:
        if inference_cfg.structure.features_key not in f:
            raise KeyError(f"Features key '{inference_cfg.structure.features_key}' not found in HDF5 file")
        # Check for labels_key, warn if missing
        if hasattr(inference_cfg.structure, 'labels_key') and inference_cfg.structure.labels_key not in f:
            warnings.warn(f"Labels key '{inference_cfg.structure.labels_key}' not found in HDF5 file", UserWarning)
            logging.warning(f"Labels key '{inference_cfg.structure.labels_key}' not found in HDF5 file")

        features = f[inference_cfg.structure.features_key]
        # Use default feature_columns if null
        feature_columns = [0, 1, 2, 3, 4, 5, 6, 21] if inference_cfg.feature_columns is None else inference_cfg.feature_columns
        if max(feature_columns) >= features.shape[1]:
            raise ValueError(
                f"Feature columns {feature_columns} exceed available features (max index: {features.shape[1] - 1})")

"""Extract total number of features from the dataset's features_key."""
"""
def extract_total_features(inference_cfg: DictConfig) -> int:
    data_path = hydra.utils.to_absolute_path(inference_cfg.data_path)
    validate_h5_structure_in_inference(data_path, inference_cfg)
    with h5py.File(data_path, 'r') as f:
        features = f[inference_cfg.structure.features_key]
        num_features = features.shape[1]  # Total features (e.g., 22)
    return num_features
"""

def load_inference_data(inference_cfg: DictConfig) -> tuple[RoboDatasetInference, np.ndarray | None]:
    """Load and preprocess unseen data for inference, optionally with labels."""
    data_path = hydra.utils.to_absolute_path(inference_cfg.data_path)
    with h5py.File(data_path, 'r') as f:
        # Load features and convert to float32 immediately
        features = f[inference_cfg.structure.features_key][:].astype(np.float32)  # Convert to float32 here

        # Default to first 8 features if feature_columns is null
        if inference_cfg.feature_columns is None:
            if features.shape[1] < 8:
                raise ValueError(f"Dataset has {features.shape[1]} features; need at least 8.")
            features = features[:, [0, 1, 2, 3, 4, 5, 6, 21]]
        else:
            features = features[:, inference_cfg.feature_columns]

        if hasattr(inference_cfg.structure, 'labels_key') and inference_cfg.structure.labels_key in f:
            labels = f[inference_cfg.structure.labels_key][:]
            #features = features[labels != 0] # skip reset
            #labels = labels[labels != 0]
            #features = features[labels != 1] # skip move
            #labels = labels[labels != 1]
            #features = features[labels != 2] # skip pick
            #labels = labels[labels != 2]
            #features = features[labels != 3] # skip carry
            #labels = labels[labels != 3]
            #features = features[labels != 4] # skip place
            #labels = labels[labels != 4]
        else:
            labels = None

    logging.info(f"Features shape: {features.shape}")

    return RoboWindowedDatasetInference(features, inference_cfg.window_size), labels