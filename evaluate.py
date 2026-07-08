"""
Evaluate trained models — metrics, confusion matrix, ROC curve.

Supports evaluating PhysioNet, FacialNet, or FusionNet independently.

Usage:
    python evaluate.py --model physio
    python evaluate.py --model facial
    python evaluate.py --model fusion
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import (set_seed, compute_metrics, plot_confusion_matrix,
                    plot_roc_curve, get_class_weights)
from data.physio_dataset import get_physio_dataloaders
from data.facial_dataset import get_facial_dataloaders
from models.physio_model import PhysioNet
from models.facial_model import FacialNet
from models.fusion_model import FusionNet


def evaluate_physio():
    """Evaluate PhysioNet on test set."""
    print("\n" + "=" * 60)
    print("  EVALUATING PhysioNet (3-Class)")
    print("=" * 60)

    _, _, test_loader, n_survey = get_physio_dataloaders(binary_mode=False)
    device = config.DEVICE

    model = PhysioNet(len(config.PHYSIO_FEATURES), n_survey, config.PHYSIO_NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(config.PHYSIO_CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for physio, survey, labels in tqdm(test_loader, desc="  Evaluating"):
            physio, survey, labels = physio.to(device), survey.to(device), labels.to(device)
            out = model(physio, survey)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    class_names = ["Calm", "Stress", "Amusement"]

    compute_metrics(all_labels, all_preds, config.PHYSIO_NUM_CLASSES, class_names)
    plot_confusion_matrix(all_labels, all_preds, class_names, "PhysioNet Confusion Matrix",
                          os.path.join(config.PLOT_DIR, "physio_confusion_matrix.png"))
    plot_roc_curve(all_labels, all_probs, config.PHYSIO_NUM_CLASSES, class_names, "PhysioNet ROC Curve",
                   os.path.join(config.PLOT_DIR, "physio_roc_curve.png"))


def evaluate_facial():
    """Evaluate FacialNet on test set."""
    print("\n" + "=" * 60)
    print("  EVALUATING FacialNet (2-Class)")
    print("=" * 60)

    _, _, test_loader = get_facial_dataloaders()
    device = config.DEVICE

    model = FacialNet(config.FACIAL_NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(config.FACIAL_CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating"):
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    class_names = ["Non-Stress", "Stress"]

    compute_metrics(all_labels, all_preds, config.FACIAL_NUM_CLASSES, class_names)
    plot_confusion_matrix(all_labels, all_preds, class_names, "FacialNet Confusion Matrix",
                          os.path.join(config.PLOT_DIR, "facial_confusion_matrix.png"))
    plot_roc_curve(all_labels, all_probs, config.FACIAL_NUM_CLASSES, class_names, "FacialNet ROC Curve",
                   os.path.join(config.PLOT_DIR, "facial_roc_curve.png"))


def evaluate_fusion():
    """Evaluate FusionNet on test set."""
    print("\n" + "=" * 60)
    print("  EVALUATING FusionNet (Binary)")
    print("=" * 60)

    from train_fusion import FusionDataset

    physio_train, physio_val, physio_test, n_survey = get_physio_dataloaders(binary_mode=True)
    facial_train, facial_val, facial_test = get_facial_dataloaders()

    fusion_test_ds = FusionDataset(physio_test.dataset, facial_test.dataset)
    test_loader = torch.utils.data.DataLoader(
        fusion_test_ds, batch_size=config.FUSION_BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True)

    device = config.DEVICE
    model = FusionNet(len(config.PHYSIO_FEATURES), n_survey,
                      config.PHYSIO_NUM_CLASSES, config.FACIAL_NUM_CLASSES,
                      config.FUSION_NUM_CLASSES, freeze_submodels=False).to(device)
    model.load_state_dict(torch.load(config.FUSION_CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for physio, survey, facial, labels in tqdm(test_loader, desc="  Evaluating"):
            physio, survey = physio.to(device), survey.to(device)
            facial, labels = facial.to(device), labels.to(device)
            out = model(physio, survey, facial)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    class_names = ["Not Stressed", "Stressed"]

    compute_metrics(all_labels, all_preds, config.FUSION_NUM_CLASSES, class_names)
    plot_confusion_matrix(all_labels, all_preds, class_names, "FusionNet Confusion Matrix",
                          os.path.join(config.PLOT_DIR, "fusion_confusion_matrix.png"))
    plot_roc_curve(all_labels, all_probs, config.FUSION_NUM_CLASSES, class_names, "FusionNet ROC Curve",
                   os.path.join(config.PLOT_DIR, "fusion_roc_curve.png"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate stress detection models")
    parser.add_argument("--model", type=str, required=True, choices=["physio", "facial", "fusion"],
                        help="Model to evaluate: physio, facial, or fusion")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)

    if args.model == "physio":
        evaluate_physio()
    elif args.model == "facial":
        evaluate_facial()
    elif args.model == "fusion":
        evaluate_fusion()


if __name__ == "__main__":
    main()
