"""
Train PhysioNet — 1D-CNN + BiLSTM on Physiological Signals.

Trains a 3-class classifier (Calm / Stress / Amusement) on the 60-second
physiological data with survey features. Uses weighted loss for class
imbalance, subject-wise splits, and early stopping.

Usage:
    python train_physio.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import EarlyStopping, set_seed, compute_metrics, plot_training_curves, get_class_weights
from data.physio_dataset import get_physio_dataloaders
from models.physio_model import PhysioNet


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="  Training", leave=False):
        physio, survey, labels = batch
        physio = physio.to(device)
        survey = survey.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(physio, survey)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * physio.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model, return average loss, accuracy, and predictions."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating", leave=False):
            physio, survey, labels = batch
            physio = physio.to(device)
            survey = survey.to(device)
            labels = labels.to(device)

            outputs = model(physio, survey)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * physio.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def main():
    print("\n" + "=" * 60)
    print("  TRAINING PhysioNet (1D-CNN + BiLSTM)")
    print("  3-Class: Calm / Stress / Amusement")
    print("=" * 60)

    # Setup
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"  Device: {device}")

    # Load data
    train_loader, val_loader, test_loader, num_survey_features = get_physio_dataloaders(binary_mode=False)

    # Compute class weights from training data
    train_labels = []
    for _, _, labels in train_loader:
        train_labels.extend(labels.numpy())
    train_labels = np.array(train_labels)
    class_weights = get_class_weights(train_labels, config.PHYSIO_NUM_CLASSES)
    print(f"\n  Class weights: {class_weights.cpu().numpy()}")

    # Create model
    model = PhysioNet(
        num_physio_channels=len(config.PHYSIO_FEATURES),
        num_survey_features=num_survey_features,
        num_classes=config.PHYSIO_NUM_CLASSES,
        dropout=config.DROPOUT_RATE,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.PHYSIO_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE
    )
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        checkpoint_path=config.PHYSIO_CHECKPOINT
    )

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n  Starting training for {config.PHYSIO_EPOCHS} epochs...\n")
    start_time = time.time()

    for epoch in range(1, config.PHYSIO_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        # LR scheduling
        scheduler.step(val_loss)

        # Record
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{config.PHYSIO_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"\n  Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        title="PhysioNet Training",
        save_path=os.path.join(config.PLOT_DIR, "physio_training_curves.png")
    )

    # ── Final Evaluation on Test Set ──
    print("\n" + "=" * 60)
    print("  EVALUATING ON TEST SET")
    print("=" * 60)

    # Load best model
    model.load_state_dict(torch.load(config.PHYSIO_CHECKPOINT, map_location=device, weights_only=True))
    _, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)

    class_names = ["Calm", "Stress", "Amusement"]
    compute_metrics(test_labels, test_preds, config.PHYSIO_NUM_CLASSES, class_names)

    print(f"\n  [OK] Best model saved at: {config.PHYSIO_CHECKPOINT}")
    print(f"  [OK] Training curves at: {os.path.join(config.PLOT_DIR, 'physio_training_curves.png')}")


if __name__ == "__main__":
    main()
