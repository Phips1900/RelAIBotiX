import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from typing import List
from sklearn.metrics import confusion_matrix


class CNNLSTM(pl.LightningModule):
    """
    def __init__(self, window_size=100, num_features=8, num_classes=4, learning_rate=0.001):
        super().__init__()
        self.window_size = window_size
        self.learning_rate = learning_rate

        # --- Convolutional Layers ---
        # In PyTorch, we expect inputs to Conv1d to be (batch, channels, seq_length).
        # Our data comes as (batch, 100, 10) so we will permute it.
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=num_features)
        # Keras used kernel_size=X_train.shape[2] (i.e. 10) for the first conv.
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.leaky_relu1 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.leaky_relu2 = nn.LeakyReLU(0.01)
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        # With an input length of 100, the convs & pooling produce:
        #  conv1: 100 - 10 + 1 = 91
        #  conv2: 91 - 3 + 1 = 89
        #  conv3: 89 - 3 + 1 = 87
        #  maxpool: floor(87/3) = 29
        # So after pooling the shape is (batch, 32, 29).

        # --- LSTM Layers ---
        # Permute output to (batch, seq_length, features) for LSTM.
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        # --- Dense / Fully-Connected Layers ---
        # Keras applies a Dense(32, relu) to each time step.
        # Here we apply a linear layer to the last dimension.
        self.fc_time = nn.Linear(50, 32)
        self.flatten = nn.Flatten()
        # After fc_time, the output has shape (batch, 29, 32), so flattening yields 29*32 = 928 features.
        # Keras then applies Dense(window_size*10)=Dense(1000) with relu.
        self.fc_dense = nn.Linear(29 * 32, window_size * 10)  # i.e. from 928 -> 1000
        # Reshape to (batch, window_size, 10) and then a final Dense to produce 7 outputs per time step.
        self.fc_final = nn.Linear(10, num_classes)

        # --- Loss function ---
        self.criterion = nn.CrossEntropyLoss()  # expects raw logits and class indices as targets

        # Store predictions and labels for confusion matrix
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):

        #Expected input x: shape (batch, window_size, num_features) e.g. (batch, 100, 10)

        # Permute to (batch, num_features, window_size) for conv layers
        x = x.permute(0, 2, 1)  # shape: (batch, 10, 100)

        # Convolutional block
        x = self.conv1(x)  # -> (batch, 128, 91)
        x = F.relu(x)
        x = self.conv2(x)  # -> (batch, 64, 89)
        x = self.leaky_relu1(x)
        x = self.conv3(x)  # -> (batch, 32, 87)
        x = self.leaky_relu2(x)
        x = self.maxpool(x)  # -> (batch, 32, 29)

        # Prepare for LSTM: permute to (batch, sequence_length, features)
        x = x.permute(0, 2, 1)  # -> (batch, 29, 32)
        x, _ = self.lstm1(x)  # -> (batch, 29, 128)
        x, _ = self.lstm2(x)  # -> (batch, 29, 50)
        x = self.dropout(x)

        # Time-distributed dense (applied to each time step)
        x = self.fc_time(x)  # -> (batch, 29, 32)

        # Flatten the sequence (29 time steps * 32 features = 928)
        x = self.flatten(x)  # -> (batch, 928)
        x = F.relu(self.fc_dense(x))  # -> (batch, window_size*10) i.e. (batch, 1000)

        # Reshape to (batch, window_size, 10)
        x = x.view(-1, self.window_size, 10)
        x = self.fc_final(x)  # -> (batch, window_size, num_classes) i.e. (batch, 100, 7)

        # Note: For CrossEntropyLoss, we leave out the softmax activation.
        return x
    """

    def __init__(self, window_size=500, num_features=8, num_classes=4, learning_rate=0.001):
        super().__init__()
        self.window_size = window_size
        self.learning_rate = learning_rate

        # --- Convolutional Layers ---
        # Match TensorFlow: first conv uses kernel_size=num_features
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=num_features)
        # Second conv: kernel_size=3, NO activation (TensorFlow has separate LeakyReLU)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.leaky_relu1 = nn.LeakyReLU(0.01)
        # Third conv: kernel_size=3, NO activation (TensorFlow has separate LeakyReLU)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.leaky_relu2 = nn.LeakyReLU(0.01)
        self.maxpool = nn.MaxPool1d(kernel_size=3)

        # Calculate sequence length after convolutions and pooling
        # With window_size=500 and num_features=8:
        # conv1: 500 - 8 + 1 = 493
        # conv2: 493 - 3 + 1 = 491
        # conv3: 491 - 3 + 1 = 489
        # maxpool: floor(489/3) = 163
        seq_len_after_conv = ((window_size - num_features + 1) - 3 + 1 - 3 + 1) // 3

        # --- LSTM Layers ---
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        # --- Dense / Fully-Connected Layers ---
        self.fc_time = nn.Linear(50, 32)
        self.flatten = nn.Flatten()

        # After fc_time and flatten: seq_len_after_conv * 32 features
        flattened_size = seq_len_after_conv * 32

        self.fc_dense = nn.Linear(flattened_size, window_size * 10)

        # Final layer
        self.fc_final = nn.Linear(10, num_classes)

        # --- Loss function ---
        self.criterion = nn.CrossEntropyLoss()

        # Store predictions and labels for confusion matrix
        self.val_preds = []
        self.val_labels = []


    def forward(self, x):
        """
        Expected input x: shape (batch, window_size, num_features) e.g. (batch, 500, 8)
        """
        # Permute to (batch, num_features, window_size) for conv layers
        x = x.permute(0, 2, 1)  # shape: (batch, 8, 500)

        # Convolutional block
        x = self.conv1(x)  # -> (batch, 128, 493)
        x = F.relu(x)

        x = self.conv2(x)  # -> (batch, 64, 491)
        x = self.leaky_relu1(x)

        x = self.conv3(x)  # -> (batch, 32, 489)
        x = self.leaky_relu2(x)

        x = self.maxpool(x)  # -> (batch, 32, 163)

        # Prepare for LSTM: permute to (batch, sequence_length, features)
        x = x.permute(0, 2, 1)  # -> (batch, 163, 32)

        # LSTM layers (both return sequences)
        x, _ = self.lstm1(x)  # -> (batch, 163, 128)
        x, _ = self.lstm2(x)  # -> (batch, 163, 50)
        x = self.dropout(x)

        # Time-distributed dense (applied to each time step)
        x = self.fc_time(x)  # -> (batch, 163, 32)
        x = F.relu(x)

        # Flatten the sequence
        x = self.flatten(x)  # -> (batch, 163*32)

        x = F.relu(self.fc_dense(x))  # -> (batch, 500*10)

        # Reshape to (batch, window_size, 10)
        x = x.view(-1, self.window_size, 10)

        # Final dense layer
        x = self.fc_final(x)  # -> (batch, 500, 7)

        # Note: For CrossEntropyLoss, we don't apply softmax here
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