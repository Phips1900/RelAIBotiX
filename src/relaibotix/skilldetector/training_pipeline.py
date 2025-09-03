import hydra
import torch
import h5py
import utils as ut
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Subset
import logging
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.model_summary import ModelSummary
import time
import numpy as np
from omegaconf import OmegaConf


class LogEpochCallback(Callback):
    """A PyTorch‑Lightning Callback that logs all metrics at the end of each epoch."""
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        if not metrics:
            self.logger.warning(f"Epoch {epoch}: Metrics not available yet.")
            return
        items = []
        for name, val in metrics.items():
            try:
                items.append(f"{name} = {float(val):.4f}")
            except Exception:
                items.append(f"{name} = {val}")
        self.logger.info(f"Epoch {epoch}: " + ", ".join(items))


@hydra.main(version_base=None, config_path="../configs", config_name="training_config.yaml")
def train(cfg: DictConfig):
    if not HydraConfig.initialized():
        raise RuntimeError("HydraConfig not initialized")
    ut.setup_logger()
    logger = logging.getLogger(__name__)
    ut.validate_h5_structure_in_train(cfg.data.path, cfg.data)
    logger.info(f"Loading data from {cfg.data.path}")

    with h5py.File(cfg.data.path, 'r') as f:
        x = f[cfg.data.structure.features_key][:, cfg.data.feature_columns]
        y = f[cfg.data.structure.labels_key][:]

    #x = x[y != 0]  # we don't classify label 0
    #y = y[y != 0]

    logger.info(f"Raw data shapes - X: {x.shape}, y: {y.shape}")
    logger.info(f"Unique labels: {np.unique(y)}")
    logger.info("Window size: {}".format(cfg.data.window_size))

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # Create the whole windowed dataset first
    full_dataset = ut.RoboWindowedDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y_encoded, dtype=torch.float32),
        cfg.data.window_size
    )

    logger.info(f"Full dataset created with {len(full_dataset)} data points")

    # Random split on window indices
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
    )

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_idx)
    logger.info(f"Training dataset size: {len(train_dataset)}")
    test_dataset = Subset(full_dataset, test_idx)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    logger.info(f"Shuffle training dataset: {cfg.data.train_shuffle}")
    logger.info(f"Shuffle test dataset: {cfg.data.test_shuffle}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.data.train_shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.data.test_shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    model = hydra.utils.instantiate(cfg.model)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=hydra.utils.instantiate(cfg.training.logger),
        callbacks=[
            hydra.utils.instantiate(cfg.training.callbacks.early_stopping),
            hydra.utils.instantiate(cfg.training.callbacks.checkpoint),
            hydra.utils.instantiate(cfg.training.callbacks.log_epoch)
        ]
    )

    summary = ModelSummary(model)
    logger.info("\n" + str(summary))
    # Train and record training time
    start_time = time.time()
    trainer.fit(model, train_loader, test_loader)
    end_time = time.time()
    training_time = end_time - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Training Complete ✅ - Time taken: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    train()
