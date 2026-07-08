"""
Central configuration for the Multimodal Stress Detection System.
Contains all hyperparameters, paths, and device settings.
"""

import os
import torch

# ──────────────────────────────────────────────
# Device Configuration
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if os.name == "nt" else 4  # Windows doesn't support multiprocessing well

# ──────────────────────────────────────────────
# Dataset Paths
# ──────────────────────────────────────────────
DATASET_ROOT = r"D:\Capstone\Stress analysis from physiological data under pressure WorkStress3D Dataset"

PHYSIO_15S_PATH = os.path.join(DATASET_ROOT, "PhysiologicalSignals", "physiological_signals_15sn.csv")
PHYSIO_30S_PATH = os.path.join(DATASET_ROOT, "PhysiologicalSignals", "physiological_signals_30sn.csv")
PHYSIO_60S_PATH = os.path.join(DATASET_ROOT, "PhysiologicalSignals", "physiological_signals_60sn.csv")

FACIAL_PATH = os.path.join(DATASET_ROOT, "TheFacialExpressions", "facial_expression.csv")

SURVEY_PSS_PATH = os.path.join(DATASET_ROOT, "Survey", "generalStressTest.csv")
SURVEY_PANAS_PATH = os.path.join(DATASET_ROOT, "Survey", "PANAS.csv")
SURVEY_DEMO_PATH = os.path.join(DATASET_ROOT, "Survey", "demographic.csv")
SURVEY_INSTANT_PATH = os.path.join(DATASET_ROOT, "Survey", "instantQuestionnaires.csv")

# ──────────────────────────────────────────────
# Checkpoint Paths
# ──────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join("D:\\Capstone", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PHYSIO_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "physio_best.pth")
FACIAL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "facial_best.pth")
FUSION_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "fusion_best.pth")

# Plot output directory
PLOT_DIR = os.path.join("D:\\Capstone", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Physiological Model Hyperparameters
# ──────────────────────────────────────────────
PHYSIO_FEATURES = ["eda", "bvp", "temp", "x", "y", "z"]  # 6 raw sensor channels
WINDOW_SIZE = 200           # Number of timesteps per sliding window
WINDOW_STRIDE = 100         # 50% overlap
PHYSIO_NUM_CLASSES = 3      # 0=Calm, 1=Stress, 2=Amusement
PHYSIO_BATCH_SIZE = 64
PHYSIO_LR = 1e-3
PHYSIO_EPOCHS = 50

# Survey feature counts (computed during dataset loading)
# PSS: 1 (TotalPoints), PANAS: 10 (C1-C10), Demographics: 6 (Gender, Age, Height, Weight, MaritalStatus, SmokingStatus)
SURVEY_FEATURE_DIM = 17     # Total survey features per subject

# ──────────────────────────────────────────────
# Facial Model Hyperparameters
# ──────────────────────────────────────────────
FACIAL_IMAGE_SIZE = 48      # 48x48 grayscale
FACIAL_NUM_CLASSES = 2      # 0=Non-Stress, 1=Stress
FACIAL_BATCH_SIZE = 8       # Small dataset → small batch
FACIAL_LR = 1e-3
FACIAL_PRETRAINED_LR = 1e-4  # Lower LR for pretrained layers
FACIAL_EPOCHS = 50
FACIAL_AUG_MULTIPLIER = 20  # 20x augmentation for 83 images

# ──────────────────────────────────────────────
# Fusion Model Hyperparameters
# ──────────────────────────────────────────────
FUSION_NUM_CLASSES = 2      # Binary: Stressed / Not Stressed
FUSION_BATCH_SIZE = 32
FUSION_LR = 1e-3
FUSION_EPOCHS = 50
FUSION_FEATURE_DIM = 256    # Feature dim from each sub-model

# ──────────────────────────────────────────────
# Training Settings
# ──────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
DROPOUT_RATE = 0.5
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Subject-wise Data Splits
# ──────────────────────────────────────────────
TRAIN_SUBJECTS = list(range(1, 15))   # Subjects 1-14
VAL_SUBJECTS = list(range(15, 18))    # Subjects 15-17
TEST_SUBJECTS = list(range(18, 21))   # Subjects 18-20


def print_config():
    """Print the current configuration."""
    print("=" * 60)
    print("MULTIMODAL STRESS DETECTION - CONFIGURATION")
    print("=" * 60)
    print(f"  Device:              {DEVICE}")
    print(f"  Dataset Root:        {DATASET_ROOT}")
    print(f"  Checkpoint Dir:      {CHECKPOINT_DIR}")
    print(f"  Random Seed:         {RANDOM_SEED}")
    print(f"  Window Size:         {WINDOW_SIZE}")
    print(f"  Physio Classes:      {PHYSIO_NUM_CLASSES}")
    print(f"  Facial Classes:      {FACIAL_NUM_CLASSES}")
    print(f"  Fusion Classes:      {FUSION_NUM_CLASSES}")
    print(f"  Train Subjects:      {TRAIN_SUBJECTS}")
    print(f"  Val Subjects:        {VAL_SUBJECTS}")
    print(f"  Test Subjects:       {TEST_SUBJECTS}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
