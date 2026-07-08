"""
Train FusionNet — Multimodal Late-Fusion for Stress Detection.

Loads pretrained PhysioNet and FacialNet weights, creates paired samples
via random within-class matching, and trains a fusion head for binary
stress classification.

Phase 1: Frozen sub-models, train fusion head only
Phase 2: Unfreeze sub-models for end-to-end fine-tuning

Usage:
    python train_fusion.py
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import EarlyStopping, set_seed, compute_metrics, plot_training_curves, get_class_weights
from data.physio_dataset import get_physio_dataloaders
from data.facial_dataset import get_facial_dataloaders
from models.fusion_model import FusionNet


class FusionDataset(Dataset):
    """Creates paired (physio, facial) samples via random within-class matching."""

    def __init__(self, physio_dataset, facial_dataset):
        self.physio_ds = physio_dataset
        self.facial_ds = facial_dataset

        # Group facial indices by label
        self.facial_by_label = {0: [], 1: []}
        for i in range(len(facial_dataset.original_images)):
            aug_mult = facial_dataset.aug_multiplier
            lbl = int(facial_dataset.original_labels[i % len(facial_dataset.original_images)])
            for j in range(aug_mult):
                idx = i + j * len(facial_dataset.original_images)
                if idx < len(facial_dataset):
                    self.facial_by_label[lbl].append(idx)

        # Ensure both classes have samples
        for lbl in [0, 1]:
            if not self.facial_by_label[lbl]:
                self.facial_by_label[lbl] = list(range(len(facial_dataset)))

    def __len__(self):
        return len(self.physio_ds)

    def __getitem__(self, idx):
        physio_window, survey, physio_label = self.physio_ds[idx]

        # Map physio label to binary: 0,2→0 (Not Stressed), 1→1 (Stressed)
        binary_label = 1 if physio_label.item() == 1 else 0

        # Get a random facial sample from the same binary class
        facial_indices = self.facial_by_label.get(binary_label, self.facial_by_label[0])
        facial_idx = facial_indices[np.random.randint(len(facial_indices))]
        facial_image, _ = self.facial_ds[facial_idx]

        return physio_window, survey, facial_image, torch.LongTensor([binary_label])[0]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for physio, survey, facial, labels in tqdm(loader, desc="  Training", leave=False):
        physio, survey = physio.to(device), survey.to(device)
        facial, labels = facial.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(physio, survey, facial)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * physio.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for physio, survey, facial, labels in tqdm(loader, desc="  Validating", leave=False):
            physio, survey = physio.to(device), survey.to(device)
            facial, labels = facial.to(device), labels.to(device)
            out = model(physio, survey, facial)
            loss = criterion(out, labels)
            total_loss += loss.item() * physio.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
              early_stopping, epochs, phase_name, device):
    """Run a training phase (frozen or unfrozen sub-models)."""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    print(f"\n  --- {phase_name}: {epochs} epochs ---\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(vl)
        train_losses.append(tl); val_losses.append(vl)
        train_accs.append(ta); val_accs.append(va)
        print(f"  Epoch {epoch:3d}/{epochs} | Train L:{tl:.4f} A:{ta:.4f} | Val L:{vl:.4f} A:{va:.4f} | {time.time()-t0:.1f}s")
        early_stopping(vl, model)
        if early_stopping.early_stop:
            print(f"\n  Early stopping at epoch {epoch}"); break

    return train_losses, val_losses, train_accs, val_accs


def main():
    print("\n" + "=" * 60)
    print("  TRAINING FusionNet (Multimodal Late Fusion)")
    print("  Binary: Stressed / Not Stressed")
    print("=" * 60)

    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"  Device: {device}")

    # Check pretrained models exist
    if not os.path.exists(config.PHYSIO_CHECKPOINT):
        print(f"  ERROR: PhysioNet checkpoint not found: {config.PHYSIO_CHECKPOINT}")
        print("  Run `python train_physio.py` first!")
        return
    if not os.path.exists(config.FACIAL_CHECKPOINT):
        print(f"  ERROR: FacialNet checkpoint not found: {config.FACIAL_CHECKPOINT}")
        print("  Run `python train_facial.py` first!")
        return

    # Load physio data (binary mode)
    physio_train, physio_val, physio_test, n_survey = get_physio_dataloaders(binary_mode=True)
    # Load facial data
    facial_train, facial_val, facial_test = get_facial_dataloaders()

    # Create fusion datasets
    fusion_train_ds = FusionDataset(physio_train.dataset, facial_train.dataset)
    fusion_val_ds = FusionDataset(physio_val.dataset, facial_val.dataset)
    fusion_test_ds = FusionDataset(physio_test.dataset, facial_test.dataset)

    train_loader = DataLoader(fusion_train_ds, batch_size=config.FUSION_BATCH_SIZE,
                               shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(fusion_val_ds, batch_size=config.FUSION_BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(fusion_test_ds, batch_size=config.FUSION_BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    print(f"  Fusion train: {len(fusion_train_ds)} | val: {len(fusion_val_ds)} | test: {len(fusion_test_ds)}")

    # Create model and load pretrained weights
    model = FusionNet(
        num_physio_channels=len(config.PHYSIO_FEATURES),
        num_survey_features=n_survey,
        physio_classes=config.PHYSIO_NUM_CLASSES,
        facial_classes=config.FACIAL_NUM_CLASSES,
        fusion_classes=config.FUSION_NUM_CLASSES,
        freeze_submodels=True,
    ).to(device)
    model.load_pretrained_weights(config.PHYSIO_CHECKPOINT, config.FACIAL_CHECKPOINT)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (Phase 1 - frozen): {trainable:,}")

    # Class weights for binary (retrieved directly to save time)
    labels_list = train_loader.dataset.physio_ds.labels
    class_weights = get_class_weights(np.array(labels_list), config.FUSION_NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Phase 1: Frozen sub-models ──
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.FUSION_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, )
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, checkpoint_path=config.FUSION_CHECKPOINT)

    start = time.time()
    tl1, vl1, ta1, va1 = run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                     early_stopping, config.FUSION_EPOCHS // 2, "Phase 1 (Frozen Sub-Models)", device)

    # ── Phase 2: End-to-end fine-tuning ──
    model.load_state_dict(torch.load(config.FUSION_CHECKPOINT, map_location=device, weights_only=True))
    model.unfreeze_submodels()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (Phase 2 - unfrozen): {trainable:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.FUSION_LR * 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, )
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, checkpoint_path=config.FUSION_CHECKPOINT)

    tl2, vl2, ta2, va2 = run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                     early_stopping, config.FUSION_EPOCHS // 2, "Phase 2 (End-to-End)", device)

    total_time = time.time() - start
    print(f"\n  Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")

    # Combined curves
    all_tl = tl1 + tl2; all_vl = vl1 + vl2; all_ta = ta1 + ta2; all_va = va1 + va2
    plot_training_curves(all_tl, all_vl, all_ta, all_va, "FusionNet Training",
                         os.path.join(config.PLOT_DIR, "fusion_training_curves.png"))

    # Test evaluation
    print("\n" + "=" * 60 + "\n  EVALUATING ON TEST SET\n" + "=" * 60)
    model.load_state_dict(torch.load(config.FUSION_CHECKPOINT, map_location=device, weights_only=True))
    _, _, tp, tlab = validate(model, test_loader, criterion, device)
    compute_metrics(tlab, tp, config.FUSION_NUM_CLASSES, ["Not Stressed", "Stressed"],
                    os.path.join(config.PLOT_DIR, "fusion_evaluation_report.txt"))
    print(f"\n  [OK] Best model: {config.FUSION_CHECKPOINT}")

if __name__ == "__main__":
    main()
