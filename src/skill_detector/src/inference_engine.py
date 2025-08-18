import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from models.cnn_lstm import CNNLSTM
from utils import load_inference_data
from models.cnn_transformer import CNNTransformer
from models.lstm_transformer import LSTMTransformer
import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names,):
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis]) * 100
        cm_percent = np.nan_to_num(cm_percent)  # replace NaN/Inf with 0

    # Create annotations with both counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)')
        annotations.append(row)

    # Plot heatmap with class names
    sns.heatmap(cm,
                annot=annotations,
                fmt='',
                cmap='Blues',
                cbar_kws={'label': 'Count'},
                xticklabels=[f'{name}' for name in class_names],
                yticklabels=[f'{name}' for name in class_names])

    plt.title(f'Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    # Show plot
    plt.show()

def plot_error_chunks(flat_labels, flat_predictions, chunk_size=5000):

    def add_label_backgrounds(ax, labels, color_map):
        prev_label = labels[0]
        start_idx = 0

        for i in range(1, len(labels)):
            if labels[i] != prev_label:
                ax.axvspan(start_idx, i, color=color_map.get(prev_label, '#ffffff'), alpha=0.1)
                start_idx = i
                prev_label = labels[i]
        # Add final span
        ax.axvspan(start_idx, len(labels), color=color_map.get(prev_label, '#ffffff'), alpha=0.1)

    def temporal_filter(labels, preds, tolerance=5):
        errors = (labels != preds)  # [True, True, True, True, False, True, False]
        clean_errors = errors.copy()

        i = 0
        while i < len(errors):
            if errors[i]:
                start = i
                while i < len(errors) and errors[i]:
                    i += 1
                end = i
                if end - start <= tolerance:
                    clean_errors[start:end] = False  # Ignore short errors
            else:
                i += 1
        return clean_errors

    color_map = {
        0: '#FFEB99',  # Medium light yellow
        1: '#FF9999',  # Medium light red
        2: '#99FF99',  # Medium light green
        3: '#9999FF',  # Medium light blue
        4: '#FFEB99',  # Medium light yellow
    }

    n = len(flat_labels)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # Slice for this chunk
        labels_chunk = flat_labels[start:end]
        preds_chunk = flat_predictions[start:end]

        # Filter misclassifications for this chunk
        filtered_errors = temporal_filter(labels_chunk, preds_chunk, tolerance=0)
        filtered_errors_as_int = filtered_errors.astype(int)

        # Plot
        fig, ax = plt.subplots(figsize=(16, 2))
        ax.plot(filtered_errors_as_int, label='Mis-classifications', color='red', linewidth=1)

        # Add background shading (chunk offset considered)
        add_label_backgrounds(ax, labels_chunk, color_map)

        ax.set_title(f"Misclassifications [{start}:{end}]")
        ax.set_xlabel("Time Steps (chunk offset)")
        ax.set_ylabel("Error")
        plt.tight_layout()
        # save to desired path and format
        #plt.savefig(f"experiments/final/dataset X/plot_{start}.png", dpi=300, bbox_inches='tight')
        plt.show()


def aggregate_predictions(sliding_preds):
    """
    Aggregates per-window predictions via majority voting to obtain one prediction per original datapoint.

    :param sliding_preds: numpy array of shape (n_windows, window_size) where each row is predictions
                          for one sliding window (one prediction per time step in the window).
    :param window_size:   The length of each sliding window (W).
    :return:              final_preds: numpy array of shape (N,) with aggregated predictions,
                          where N = n_windows + window_size - 1.
    """
    n_windows, W = sliding_preds.shape
    N = n_windows + W - 1  # total number of original datapoints
    final_preds = np.empty(N, dtype=sliding_preds.dtype)

    # For each original datapoint j, collect predictions from all windows that cover it.
    for j in range(N):
        i_start = max(0, j - W + 1)
        i_end = min(j, n_windows - 1)
        # For each window i covering datapoint j, the prediction for datapoint j is at position (j - i) in the window.
        votes = [sliding_preds[i, j - i] for i in range(i_start, i_end + 1)]
        # Majority vote: count occurrences and choose the label with highest count.
        final_preds[j] = np.argmax(np.bincount(votes))

    return final_preds


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        logging.info(f"Device '{device_str}' unavailable; using 'cpu'.")
        return torch.device("cpu")

def load_model(model_type: str, checkpoint_path: str, window_size: int, num_features: int, num_classes:int):
    if model_type == 'cnn_transformer':
        model = CNNTransformer(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    elif model_type == 'cnn_lstm':
        model = CNNLSTM(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    elif model_type == 'lstm_transformer':
        model = LSTMTransformer(window_size=window_size, num_features=num_features, num_classes=num_classes).float()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f"Loaded {model_type} from {checkpoint_path}")
    return model

"""
def model_select_by_num(num_features: int, selector_cfg: DictConfig) -> tuple[str, str]:
    if selector_cfg.type == "rule_based":
        rules = selector_cfg.rules
        for rule in rules:
            if num_features == rule.num_features:
                return rule.checkpoint, rule.model
        return selector_cfg.rule_based.default.checkpoint, selector_cfg.rule_based.default.model
    else:
        raise ValueError(f"Unknown model_selector type: {selector_cfg.type}")
"""

def model_select_by_type(robot_type: str, selector_cfg: DictConfig) -> tuple[str, str]:
    if selector_cfg.type == "rule_based":
        for rule in selector_cfg.rules:
            if rule.robot == robot_type['robot_type']:
                return rule.checkpoint, rule.model
        raise ValueError(f"Unknown robot type.")
    else:
        raise ValueError(f"Unknown model_selector type: {selector_cfg.type}")

@hydra.main(config_path="../configs", config_name="inference_config", version_base=None)
def infer(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    #logger.info(f"Extracting features from {cfg.data_path}...")
    #num_features_total = extract_total_features(cfg)

    logger.info(f"Running inference on {cfg.data_path}...")

    feature_columns = cfg.feature_columns
    #checkpoint_path, model_type = model_select_by_num(num_features_total, cfg.model_selector)
    checkpoint_path, model_type = model_select_by_type(cfg.robot, cfg.model_selector)

    model = load_model(model_type, checkpoint_path, cfg.window_size, len(feature_columns), cfg.num_classes)
    model.eval()
    model.to(device)

    dataset, labels = load_inference_data(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # Collect all sliding window predictions
    all_window_preds = []
    inference_start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.float().to(device)
            logits = model(batch)  # [B, W, C]
            timestep_preds = torch.argmax(logits, dim=-1)  # [B, W]
            all_window_preds.extend(timestep_preds.cpu().numpy())  # shape: [n_windows, window_size]

    inference_end = time.time()
    total_inference_time = inference_end - inference_start
    logger.info(f"Total inference time: {total_inference_time:.4f} seconds")
    logger.info(f"Average time per batch: {total_inference_time / len(dataloader):.6f} seconds")

    all_window_preds = np.array(all_window_preds)  # Shape: [n_windows, window_size]

    # Aggregate predictions per timestep
    flat_predictions = aggregate_predictions(all_window_preds)

    logger.info(
        f"Inference complete."
    )

    if labels is not None:
        flat_labels = labels.flatten()
        flat_predictions = flat_predictions # Adjust predictions to match label indexing
        print("Unique labels:", np.unique(flat_labels))
        print("Unique predictions:", np.unique(flat_predictions))
        print(f"Total predictions: {len(flat_predictions)}")
        print(f"Total labels: {len(flat_labels)}")
        print(f"Window size: {cfg.window_size}")
        print(f"First 20 predictions: {flat_predictions[:20]}")
        print(f"First 20 labels: {flat_labels[:20]}")

        # Metrics
        acc = accuracy_score(flat_labels, flat_predictions)
        f1 = f1_score(flat_labels, flat_predictions, average='weighted')
        report = classification_report(flat_labels, flat_predictions, zero_division=0)
        print("Classification Report:\n", report)
        logger.info(f"Accuracy before post-processing: {acc:.5f}")
        logger.info(f"F1 Score before post-processing: {f1:.5f}")

        # Confusion matrix info and visualization
        cm = confusion_matrix(flat_labels, flat_predictions)
        print(f"\nConfusion Matrix Shape: {cm.shape}")
        #class_names = ['idle', 'move', 'pick', 'carry', 'place' 'rotate', 'shake', 'pour'] #alle_skills
        class_names = ['reset', 'move', 'pick', 'carry', 'place']  # pick_and_place
        plot_confusion_matrix(cm, class_names)

        # Plot chunk-wise misclassifications between skills, useful for error visualization
        plot_error_chunks(flat_labels, flat_predictions, chunk_size=20000)
        return flat_predictions, model_type, acc, f1

if __name__ == "__main__":
    infer()
