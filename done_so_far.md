# ✅ Multimodal Stress Detection System — Work Done So Far

> **Project:** Capstone — Multimodal Stress Detection using Physiological Signals & Facial Expressions  
> **Dataset:** WorkStress3D (20 subjects, physiological signals + facial expressions + surveys)  
> **Framework:** PyTorch  
> **Last Updated:** 2026-05-09

---

## 📁 Project Structure

```
D:\Capstone\
├── config.py                  # Central configuration (hyperparams, paths, device)
├── utils.py                   # EarlyStopping, metrics, plotting helpers
├── requirements.txt           # Python dependencies
├── _fix_unicode.py            # Unicode fix utility for Windows
│
├── data/
│   ├── __init__.py
│   ├── physio_dataset.py      # Physiological signal data loading + sliding windows
│   └── facial_dataset.py      # Facial expression data loading + augmentation
│
├── models/
│   ├── __init__.py
│   ├── physio_model.py        # PhysioNet (1D-CNN + BiLSTM)
│   ├── facial_model.py        # FacialNet (ResNet-18 Transfer Learning)
│   └── fusion_model.py        # FusionNet (Late Fusion)
│
├── train_physio.py            # PhysioNet training script
├── train_facial.py            # FacialNet training script
├── train_fusion.py            # FusionNet two-phase training script
├── evaluate.py                # Unified evaluation script (metrics, confusion matrix, ROC)
│
├── checkpoints/
│   ├── physio_best.pth        # Best PhysioNet weights (~2.9 MB)
│   ├── facial_best.pth        # Best FacialNet weights (~45 MB)
│   └── fusion_best.pth        # Best FusionNet weights (~49 MB)
│
├── plots/
│   ├── physio_training_curves.png
│   ├── physio_confusion_matrix.png
│   ├── physio_roc_curve.png
│   ├── facial_training_curves.png
│   ├── facial_confusion_matrix.png
│   ├── facial_roc_curve.png
│   ├── fusion_training_curves.png
│   ├── fusion_confusion_matrix.png
│   └── fusion_roc_curve.png
│
└── Stress analysis from physiological data under pressure WorkStress3D Dataset/
    ├── PhysiologicalSignals/   (15s, 30s, 60s CSV files)
    ├── TheFacialExpressions/   (facial_expression.csv — 83 images as pixel strings)
    └── Survey/                 (PSS, PANAS, Demographics, Instant Questionnaires)
```

---

## 1️⃣ Configuration (`config.py`)

| Setting | Value |
|---------|-------|
| **Device** | CUDA (GPU) if available, else CPU |
| **Random Seed** | 42 |
| **Data Split** | Subject-wise: Train (1–14), Val (15–17), Test (18–20) |
| **Early Stopping Patience** | 7 epochs |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) |
| **Dropout Rate** | 0.5 |

### PhysioNet Hyperparameters
| Parameter | Value |
|-----------|-------|
| Features | EDA, BVP, Temperature, Accel X/Y/Z (6 channels) |
| Window Size | 200 timesteps |
| Window Stride | 100 (50% overlap) |
| Classes | 3 (Calm, Stress, Amusement) |
| Batch Size | 64 |
| Learning Rate | 1e-3 |
| Epochs | 50 |
| Survey Features | 17 (PSS + PANAS + Demographics) |

### FacialNet Hyperparameters
| Parameter | Value |
|-----------|-------|
| Image Size | 48×48 grayscale |
| Classes | 2 (Non-Stress, Stress) |
| Batch Size | 8 |
| Learning Rate (Head) | 1e-3 |
| Learning Rate (Pretrained) | 1e-4 |
| Epochs | 50 |
| Augmentation Multiplier | 20× (for 83 images) |

### FusionNet Hyperparameters
| Parameter | Value |
|-----------|-------|
| Classes | 2 (Stressed, Not Stressed) |
| Batch Size | 32 |
| Learning Rate | 1e-3 (Phase 1), 1e-4 (Phase 2) |
| Epochs | 50 (25 per phase) |
| Feature Dimension | 256 per sub-model (512 concatenated) |

---

## 2️⃣ Data Pipeline (Completed ✅)

### Physiological Dataset (`data/physio_dataset.py`)
- **Survey Feature Loading**: Loads PSS (1 feature), PANAS (10 features), Demographics (6 features) = **17 total survey features** per subject
- **Categorical Encoding**: Gender, MaritalStatus, SmokingStatus encoded with `LabelEncoder`
- **Normalization**: StandardScaler on all survey and physiological features
- **Sliding Windows**: Creates overlapping windows (200 timesteps, 50% overlap) from 60-second physiological data
- **Subject-wise Split**: Prevents data leakage — no subject appears in multiple splits
- **Output Format**: `(batch, 6, window_size)` physio + `(batch, 17)` survey → label

### Facial Dataset (`data/facial_dataset.py`)
- **Data Source**: 83 facial expression images stored as pixel strings in CSV
- **Pixel Parsing**: Converts pixel strings to 48×48 numpy arrays
- **Augmentation (20×)**: RandomHorizontalFlip, RandomRotation(15°), RandomAffine (translate, scale, shear), ColorJitter — applied to training set only
- **Stratified Split**: 70/15/15 train/val/test (preserves class distribution)
- **Output Format**: `(batch, 1, 48, 48)` grayscale image → label

---

## 3️⃣ Model Architectures (Completed ✅)

### PhysioNet (`models/physio_model.py`)
```
Architecture: Conv1D ×3 → BiLSTM → concat with survey → FC → output

Input: (batch, 6, 200) physio + (batch, 17) survey
  ↓
Conv1D Block 1: 6→32 ch, kernel=7, BN, ReLU, MaxPool, Dropout(0.3)
Conv1D Block 2: 32→64 ch, kernel=5, BN, ReLU, MaxPool, Dropout(0.3)
Conv1D Block 3: 64→128 ch, kernel=3, BN, ReLU, MaxPool, Dropout(0.3)
  ↓
BiLSTM: 128→128 hidden, 2 layers, bidirectional → 256-d output
  ↓
Concat: 256 (BiLSTM) + 17 (survey) = 273
  ↓
FC: 273→128 → ReLU → Dropout(0.5) → 128→3 (logits)
  ↓
Output: (batch, 3)  — Calm / Stress / Amusement
```
- **Feature extraction method**: `get_features()` returns 256-d embedding (without survey) for fusion

### FacialNet (`models/facial_model.py`)
```
Architecture: Modified ResNet-18 (pretrained) → custom head

Input: (batch, 1, 48, 48) grayscale
  ↓
Conv1 (modified): 1-channel input (mean of pretrained 3-ch weights)
BN1 → ReLU → MaxPool
  ↓
Layer 1 (FROZEN) → Layer 2 (FROZEN) → Layer 3 (fine-tuned) → Layer 4 (fine-tuned)
  ↓
AdaptiveAvgPool → Flatten → 512-d
  ↓
Feature Layer: 512→256, ReLU, Dropout(0.5) → 256-d embedding
  ↓
Classifier: 256→2 (logits)
  ↓
Output: (batch, 2)  — Non-Stress / Stress
```
- **Transfer Learning**: Pretrained ResNet-18 with frozen early layers (1-2), fine-tuned later layers (3-4)
- **Differential LR**: Lower LR (1e-4) for pretrained, higher LR (1e-3) for new head
- **Feature extraction method**: `get_features()` returns 256-d embedding for fusion

### FusionNet (`models/fusion_model.py`)
```
Architecture: Late Fusion — concatenate sub-model features → fusion head

PhysioNet → get_features() → 256-d ┐
                                     ├→ Concat → 512-d
FacialNet → get_features() → 256-d ┘
  ↓
Fusion Head:
  512→256, BN, ReLU, Dropout(0.5)
  256→64, ReLU
  64→2 (logits)
  ↓
Output: (batch, 2)  — Not Stressed / Stressed
```
- **Pretrained Sub-models**: Loads PhysioNet and FacialNet checkpoints
- **Freeze/Unfreeze**: Supports freezing sub-models (Phase 1) and unfreezing for end-to-end fine-tuning (Phase 2)
- **Smart Unfreezing**: When unfreezing, keeps ResNet early layers (1-2) frozen

---

## 4️⃣ Training Scripts (Completed ✅)

### PhysioNet Training (`train_physio.py`)
- Weighted CrossEntropyLoss for class imbalance
- Adam optimizer with ReduceLROnPlateau
- EarlyStopping (patience=7)
- Saves best model to `checkpoints/physio_best.pth`
- Generates training curves plot
- Final test evaluation with classification report

### FacialNet Training (`train_facial.py`)
- Differential learning rates via `model.get_param_groups()`
- Weighted CrossEntropyLoss for class imbalance
- EarlyStopping (patience=7)
- Saves best model to `checkpoints/facial_best.pth`
- Generates training curves plot
- Final test evaluation with classification report

### FusionNet Training (`train_fusion.py`)
- **Two-Phase Training Strategy**:
  - **Phase 1 (25 epochs)**: Sub-models frozen, only fusion head trains → LR=1e-3
  - **Phase 2 (25 epochs)**: Sub-models unfrozen for end-to-end fine-tuning → LR=1e-4
- **Random Within-Class Matching**: Creates paired (physio, facial) samples by matching binary labels
  - Physio labels mapped: Calm/Amusement→0 (Not Stressed), Stress→1 (Stressed)
- Custom `FusionDataset` class for pairing modalities
- Combined training curves plotted across both phases

---

## 5️⃣ Evaluation Pipeline (`evaluate.py`) (Completed ✅)

- **CLI-based**: `python evaluate.py --model {physio|facial|fusion}`
- **For each model generates**:
  - Accuracy, Precision, Recall, F1 Score (weighted)
  - Full Classification Report
  - Confusion Matrix Heatmap (saved as PNG)
  - ROC Curve with per-class AUC (saved as PNG)

---

## 6️⃣ Utilities (`utils.py`) (Completed ✅)

| Utility | Description |
|---------|-------------|
| `EarlyStopping` | Halts training when val loss stops improving; saves best checkpoint |
| `set_seed()` | Sets random seed across numpy, torch, CUDA for reproducibility |
| `compute_metrics()` | Prints accuracy, precision, recall, F1 + full classification report |
| `plot_training_curves()` | Plots train/val loss and accuracy curves side-by-side |
| `plot_confusion_matrix()` | Plots confusion matrix as a seaborn heatmap |
| `plot_roc_curve()` | Plots per-class ROC curves with AUC scores |
| `get_class_weights()` | Computes inverse-frequency weights for imbalanced data |

---

## 7️⃣ Trained Models & Results (Completed ✅)

### Saved Checkpoints
| Model | File | Size |
|-------|------|------|
| PhysioNet | `checkpoints/physio_best.pth` | ~2.9 MB |
| FacialNet | `checkpoints/facial_best.pth` | ~45 MB |
| FusionNet | `checkpoints/fusion_best.pth` | ~49 MB |

### Generated Plots (9 total)
| Model | Training Curves | Confusion Matrix | ROC Curve |
|-------|----------------|-------------------|-----------|
| PhysioNet | ✅ `physio_training_curves.png` | ✅ `physio_confusion_matrix.png` | ✅ `physio_roc_curve.png` |
| FacialNet | ✅ `facial_training_curves.png` | ✅ `facial_confusion_matrix.png` | ✅ `facial_roc_curve.png` |
| FusionNet | ✅ `fusion_training_curves.png` | ✅ `fusion_confusion_matrix.png` | ✅ `fusion_roc_curve.png` |

---

## 8️⃣ Dependencies (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
opencv-python>=4.8.0
```

---

## 9️⃣ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train PhysioNet (3-class: Calm/Stress/Amusement)
python train_physio.py

# 3. Train FacialNet (2-class: Non-Stress/Stress)
python train_facial.py

# 4. Train FusionNet (requires both models trained first)
python train_fusion.py

# 5. Evaluate any model
python evaluate.py --model physio
python evaluate.py --model facial
python evaluate.py --model fusion
```

---

## 🔑 Key Design Decisions

1. **Subject-wise splitting** — Prevents data leakage; no subject appears in train+test
2. **Survey feature enrichment** — PSS, PANAS, and demographic features provide subject-level context beyond raw signals
3. **Late fusion** — Concatenates learned feature embeddings rather than raw data, allowing each modality to learn independently
4. **Two-phase fusion training** — First trains only the fusion head (stable features), then fine-tunes end-to-end (joint optimization)
5. **20× augmentation** — Compensates for the tiny 83-image facial dataset
6. **Transfer learning with frozen layers** — ResNet-18 early layers frozen to prevent overfitting on small data
7. **Differential learning rates** — Pretrained layers get 10× lower LR than new layers
8. **Weighted loss** — Handles class imbalance via inverse-frequency class weights
9. **60-second window** — Uses longest available physiological recording for maximum temporal context

---

## ✅ Summary — Everything Completed

| Component | Status |
|-----------|--------|
| Project Structure & Config | ✅ Done |
| Physiological Data Pipeline (survey + sliding windows) | ✅ Done |
| Facial Expression Data Pipeline (augmentation + split) | ✅ Done |
| PhysioNet Model (1D-CNN + BiLSTM) | ✅ Done |
| FacialNet Model (ResNet-18 Transfer Learning) | ✅ Done |
| FusionNet Model (Late Fusion) | ✅ Done |
| PhysioNet Training + Checkpoint | ✅ Done |
| FacialNet Training + Checkpoint | ✅ Done |
| FusionNet Two-Phase Training + Checkpoint | ✅ Done |
| Evaluation Pipeline (metrics, confusion matrix, ROC) | ✅ Done |
| Training Curves Plots | ✅ Done |
| Confusion Matrix Plots | ✅ Done |
| ROC Curve Plots | ✅ Done |
| All 3 Model Checkpoints Saved | ✅ Done |
