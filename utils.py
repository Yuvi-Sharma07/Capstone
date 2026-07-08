"""
Utility functions for training, evaluation, and visualization.
Includes EarlyStopping, metric computation, and plotting helpers.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import config


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience=7, min_delta=0.0, checkpoint_path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Save model when validation loss decreases."""
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"  [OK] Model saved to {self.checkpoint_path} (val_loss: {self.best_loss:.4f})")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred, num_classes, class_names=None, save_path=None):
    """Compute classification metrics and optionally save them to a file."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    output = []
    output.append(f"\n{'='*50}")
    output.append(f"  Accuracy:  {acc:.4f}")
    output.append(f"  Precision: {prec:.4f}")
    output.append(f"  Recall:    {rec:.4f}")
    output.append(f"  F1 Score:  {f1:.4f}")
    output.append(f"{'='*50}")

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    output.append("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    output.append(report)

    output_str = "\n".join(output)
    print(output_str)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"  [OK] Evaluation report saved to {save_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, title="Training", save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", markersize=3, label="Val Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, train_accs, "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, val_accs, "r-o", markersize=3, label="Val Acc")
    ax2.set_title(f"{title} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="gray"
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_probs, num_classes, class_names=None, title="ROC Curve", save_path=None):
    """Plot ROC curve with AUC for each class."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])

    plt.figure(figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] ROC curve saved to {save_path}")
    plt.close()


def get_class_weights(labels, num_classes):
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    # Avoid division by zero
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # Normalize
    return torch.FloatTensor(weights).to(config.DEVICE)
