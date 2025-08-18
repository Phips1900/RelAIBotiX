import torch
import torch.nn as nn
import math
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
from sklearn.metrics import confusion_matrix

class LearnableAbsolutePositionalEncoding(nn.Module):
    r"""Implements learnable positional encodings to capture the relative or absolute position of
    tokens in a sequence. Unlike fixed positional encodings (e.g., sine/cosine), these encodings
    are initialized randomly and optimized during training, allowing the model to adapt positional
    information to the specific task. The encodings are added to input embeddings and followed by
    dropout.

    The positional encoding for each position is a learnable vector of size `d_model`, initialized
    uniformly in the range [-0.02, 0.02]. The encodings are stored as a parameter and updated via
    backpropagation.

    Args:
        d_model (int): The dimension of the input embeddings (required).
        dropout (float, optional): The dropout probability applied after adding positional encodings.
            Default: 0.1.
        max_len (int, optional): The maximum sequence length for which to precompute positional
            encodings. Default: 5000.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied to the sum of input and positional encodings.
        pe (nn.Parameter): Learnable positional encoding matrix of shape [1, max_len, d_model],
            optimized during training.

    Example:
        >>> pos_encoder = LearnableAbsolutePositionalEncoding(d_model=128, dropout=0.1, max_len=100)
        >>> x = torch.randn(32, 100, 128)  # [batch_size, seq_len, d_model]
        >>> x = pos_encoder(x)  # Output: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(LearnableAbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding, stored as a learnable parameter
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # Shape: [1, max_len, d_model]
        nn.init.uniform_(self.pe, -0.02, 0.02)  # Initialize uniformly between -0.02 and 0.02

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Adds learnable positional encodings to the input tensor and applies dropout.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model], with positional
                encodings added and dropout applied.

        Note:
            The positional encoding matrix is sliced to match the input sequence length (`seq_len`).
            Ensure `seq_len <= max_len` to avoid indexing errors.
        """
        # Slice positional encodings to match the input sequence length
        x = x + self.pe[:, :x.size(1), :]  # pe: [1, seq_len, d_model]
        return self.dropout(x)

class FixedAbsolutePositionalEncoding(nn.Module):
    r"""Implements fixed positional encoding based on sine and cosine functions, as introduced in
    'Attention is All You Need' (Vaswani et al., 2017). Adds positional encodings to input embeddings
    to capture the relative or absolute position of tokens in a sequence. The encodings are computed
    using sinusoidal functions and are non-trainable, stored as a buffer.

    The positional encoding for position `pos` and dimension `i` is defined as:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        d_model (int): The dimension of the input embeddings (required).
        dropout (float, optional): The dropout probability applied after adding positional encodings.
            Default: 0.1.
        max_len (int, optional): The maximum sequence length for which to precompute positional
            encodings. Default: 5000.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied to the sum of input and positional encodings.
        pe (torch.Tensor): Precomputed positional encoding matrix of shape [1, max_len, d_model],
            stored as a non-trainable buffer.

    Example:
        >>> pos_encoder = FixedAbsolutePositionalEncoding(d_model=128, dropout=0.1, max_len=100)
        >>> x = torch.randn(32, 100, 128)  # [batch_size, seq_len, d_model]
        >>> x = pos_encoder(x)  # Output: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(FixedAbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Adds positional encodings to the input tensor and applies dropout.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model], with positional
                encodings added and dropout applied.

        Note:
            The positional encoding matrix is sliced to match the input sequence length (`seq_len`).
            Ensure `seq_len <= max_len` to avoid indexing errors.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CNNTransformer(pl.LightningModule):
    def __init__(
        self,
        window_size: int = 100,
        num_features: int = 8,
        num_classes: int = 5,
        learning_rate: float = 0.001,
        cnn_channels: list = [64, 128],
        transformer: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.window_size = window_size
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Unpack transformer config
        tcfg = transformer or {}
        encoding = tcfg.get("encoding", "learnable")
        d_model   = tcfg.get("d_model", cnn_channels[-1])
        nhead     = tcfg.get("nhead", 8)
        num_layers= tcfg.get("num_layers", 1)
        dropout   = tcfg.get("dropout", 0.1)

        # Build CNN dynamically
        layers = []
        in_ch = num_features
        for out_ch in cnn_channels:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = out_ch
        self.cnn_layers = nn.Sequential(*layers)

        # Positional encoder & transformer
        if encoding == "fixed":
            self.pos_encoder = FixedAbsolutePositionalEncoding(d_model, dropout, max_len=window_size)
        elif encoding == "learnable":
            self.pos_encoder = LearnableAbsolutePositionalEncoding(d_model, dropout, max_len=window_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        self.fc = nn.Linear(d_model, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        logits = self.fc(x)
        return logits

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
        #class_names = ['move', 'grab', 'carry', 'place', 'reset', 'rotate', 'shake', 'pour'] #alle_skills
        class_names = ['idle', 'move', 'pick', 'carry', 'place']  # pick_and_place
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