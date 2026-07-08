"""
Train FacialNet — ResNet-18 Transfer Learning for Facial Expressions.

Trains a 2-class classifier (Stress / Non-Stress) on 83 facial images
with 20x data augmentation. Uses differential learning rates, weighted loss,
and early stopping.

Usage:
    python train_facial.py
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import EarlyStopping, set_seed, compute_metrics, plot_training_curves, get_class_weights
from data.facial_dataset import get_facial_dataloaders
from models.facial_model import FacialNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    print("\n" + "=" * 60)
    print("  TRAINING FacialNet (ResNet-18 Transfer Learning)")
    print("  2-Class: Stress / Non-Stress")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"  Device: {device}")

    train_loader, val_loader, test_loader = get_facial_dataloaders()

    # Class weights (retrieved directly from dataset to save time)
    train_labels = train_loader.dataset.original_labels
    class_weights = get_class_weights(np.array(train_labels), config.FACIAL_NUM_CLASSES)
    print(f"\n  Class weights: {class_weights.cpu().numpy()}")

    model = FacialNet(num_classes=config.FACIAL_NUM_CLASSES, dropout=config.DROPOUT_RATE).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_p:,} | Trainable: {train_p:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.get_param_groups(config.FACIAL_PRETRAINED_LR, config.FACIAL_LR))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, checkpoint_path=config.FACIAL_CHECKPOINT)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    print(f"\n  Starting training for {config.FACIAL_EPOCHS} epochs...\n")
    start = time.time()

    for epoch in range(1, config.FACIAL_EPOCHS + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, vp, vlab = validate(model, val_loader, criterion, device)
        scheduler.step(vl)
        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta); val_accs.append(va)
        print(f"  Epoch {epoch:3d}/{config.FACIAL_EPOCHS} | Train L:{tl:.4f} A:{ta:.4f} | Val L:{vl:.4f} A:{va:.4f} | {time.time()-t0:.1f}s")
        early_stopping(vl, model)
        if early_stopping.early_stop:
            print(f"\n  Early stopping at epoch {epoch}"); break

    print(f"\n  Training completed in {time.time()-start:.1f}s")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, "FacialNet Training",
                         os.path.join(config.PLOT_DIR, "facial_training_curves.png"))

    # Test evaluation
    print("\n" + "=" * 60 + "\n  EVALUATING ON TEST SET\n" + "=" * 60)
    model.load_state_dict(torch.load(config.FACIAL_CHECKPOINT, map_location=device, weights_only=True))
    _, _, tp, tl = validate(model, test_loader, criterion, device)
    compute_metrics(tl, tp, config.FACIAL_NUM_CLASSES, ["Non-Stress", "Stress"],
                    os.path.join(config.PLOT_DIR, "facial_evaluation_report.txt"))
    print(f"\n  [OK] Best model: {config.FACIAL_CHECKPOINT}")

if __name__ == "__main__":
    main()
