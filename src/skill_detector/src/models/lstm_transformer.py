import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from typing import List
from sklearn.metrics import confusion_matrix

class LSTMTransformer(pl.LightningModule):
    def __init__(self, window_size=100, num_features=8, num_classes=4, learning_rate=0.001):
        super().__init__()
        self.window_size = window_size
        self.learning_rate = learning_rate

        # Unidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            batch_first=True,
            bidirectional=False,
            num_layers=1
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=False,
            num_layers=1
        )

        # Transformer encoder for attention refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.dropout = nn.Dropout(0.2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.val_preds = []
        self.val_labels = []


    def forward(self, x):
        # Input shape: (batch, window_size, num_features)

        # Unidirectional LSTM processing (preserves causality)
        x, _ = self.lstm1(x)  # (batch, seq_len, 128)
        x, _ = self.lstm2(x)  # (batch, seq_len, 128)

        # Transformer encoder (bidirectional attention)
        x = self.transformer_encoder(x)  # (batch, seq_len, 128)

        x = self.dropout(x)

        # Per-timestep classification
        x = self.classifier(x)  # (batch, seq_len, num_classes)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_indices = torch.argmax(y, dim=-1)
        logits = logits.view(-1, logits.shape[-1])
        y_indices = y_indices.view(-1)
        loss = self.criterion(logits, y_indices)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_indices = torch.argmax(y, dim=-1)
        logits = logits.view(-1, logits.shape[-1])
        y_indices = y_indices.view(-1)
        loss = self.criterion(logits, y_indices)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y_indices).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.val_preds.extend(preds.detach().cpu().numpy().tolist())
        self.val_labels.extend(y_indices.detach().cpu().numpy().tolist())
        return loss

    def plot_confusion_matrix(self,
                              cm: np.ndarray,
                              class_names: List[str] = None,
                              show_raw: bool = False,
                              show_row_norm: bool = True,
                              show_col_norm: bool = True,
                              epoch: int = 0,
                              logger=None,
                              base_tag: str = "Confusion Matrix"
                              ):
        # Raw CM as a separate figure
        if show_raw:
            fig_raw, ax_raw = plt.subplots(figsize=(6, 5))
            raw_mat = np.nan_to_num(cm)
            sns.heatmap(
                raw_mat,
                annot=True, fmt="d",
                cmap="Blues",
                ax=ax_raw,
                cbar=True
            )
            ax_raw.set_title("Raw counts")
            ax_raw.set_xlabel("Predicted")
            ax_raw.set_ylabel("True")
            if class_names:
                ax_raw.set_xticks(range(len(class_names)))
                ax_raw.set_yticks(range(len(class_names)))
                ax_raw.set_xticklabels(class_names, rotation=45, ha='right')
                ax_raw.set_yticklabels(class_names, rotation=0)

            plt.tight_layout()
            if logger:
                logger.experiment.add_figure(f"{base_tag}_raw", fig_raw, epoch)
            plt.close(fig_raw)

        # Row- and column-normalized CM in a combined figure
        modes = []
        with np.errstate(invalid='ignore', divide='ignore'):
            row_norm = cm.astype(float) / cm.sum(1, keepdims=True)
            col_norm = cm.astype(float) / cm.sum(0, keepdims=True)


        if show_row_norm:
            modes.append(("Row-norm (recall)", row_norm, ".4f"))
        if show_col_norm:
            modes.append(("Col-norm (precision)", col_norm, ".4f"))

        if modes:
            n = len(modes)
            fig_norm, axes = plt.subplots(1, n, figsize=(6 * n, 5))
            if n == 1:
                axes = [axes]

            for ax, (title, mat, fmt) in zip(axes, modes):
                mat = np.nan_to_num(mat)
                sns.heatmap(
                    mat,
                    annot=True, fmt=fmt,
                    cmap="Blues",
                    ax=ax,
                    cbar=True
                )
                ax.set_title(title)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                if class_names:
                    ax.set_xticks(range(len(class_names)))
                    ax.set_yticks(range(len(class_names)))
                    ax.set_xticklabels(class_names, rotation=45, ha='right')
                    ax.set_yticklabels(class_names, rotation=0)

            plt.tight_layout()
            if logger:
                logger.experiment.add_figure(f"{base_tag}_norm", fig_norm, epoch)
            plt.close(fig_norm)

    def on_validation_epoch_end(self):
        cm = confusion_matrix(self.val_labels, self.val_preds)
        # class_names = ['move', 'grab', 'carry', 'place', 'reset', 'rotate', 'shake', 'pour'] #alle_skills_v1.py
        class_names = ['move', 'pick', 'carry', 'place']  # pick_and_place_trials.py
        # draw row- and col-normalized side by side
        self.plot_confusion_matrix(
            cm,
            class_names=class_names,
            show_raw=True,
            show_row_norm=True,
            show_col_norm=True,
            epoch=self.current_epoch,
            logger=self.logger,
            base_tag="CM_comparison"
        )
        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        return torch.optim.Adamax(self.parameters(), lr=self.learning_rate)