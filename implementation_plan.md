# Multimodal Stress Detection — DL Training Pipeline

Train deep learning models on the **WorkStress3D dataset** to detect stress by fusing physiological signals (EDA, BVP, temperature, accelerometer) with facial expression analysis. Trained models will later be used for real-time inference with wearable glove.

---

## Dataset Analysis (New Dataset)

### Physiological Signals — 20 subjects, 3 time windows

| File | Samples | Columns | Class 0 (Calm) | Class 1 (Stress) | Class 2 (Amusement) |
|------|---------|---------|----------------|-------------------|---------------------|
| `physiological_signals_15sn.csv` | 39,200 × 8 | Subject, eda, bvp, temp, x, y, z, emotion | 25,704 | 13,272 | 224 |
| `physiological_signals_30sn.csv` | 82,600 × 8 | subjet, eda, bvp, temp, x, y, z, emotion | 54,162 | 27,966 | 472 |
| `physiological_signals_60sn.csv` | 168,000 × 8 | Subject, eda, bvp, temp, x, y, z, emotion | 110,160 | 56,880 | 960 |

**Features**: EDA (skin conductance), BVP (blood volume pulse), Temperature, Accelerometer (x, y, z)  
**Subjects**: 20 participants  

> [!WARNING]
> **Severe class imbalance**: ~65% Calm, ~34% Stress, ~0.6% Amusement. Will use weighted loss + oversampling.

### Facial Expression Data

| File | Samples | Format |
|------|---------|--------|
| `facial_expression.csv` | 83 × 2 | emotion (0/1), pixels (48×48 = 2304 values) |

- **Only 83 images** — 45 non-stress (0), 38 stress (1)
- 48×48 grayscale images stored as pixel strings
- No train/test split column

> [!IMPORTANT]
> **83 samples is very small** for training a CNN from scratch. We'll use heavy augmentation + transfer learning with a frozen pretrained backbone to avoid overfitting.

### Survey Data

| File | Records | Purpose |
|------|---------|---------|
| `generalStressTest.csv` | 20 | Perceived Stress Scale per subject (TotalPoints) |
| `PANAS.csv` | 20 | Positive/Negative Affect Schedule per subject |
| `instantQuestionnaires.csv` | 651 | Momentary emotional states (Neutral, Tension, Happy, etc.) |
| `demographic.csv` | 20 | Subject demographics |

---

## Proposed Architecture

```
┌──────────────────────────┐    ┌──────────────────────────┐
│   PhysioNet              │    │   FacialNet              │
│   (1D-CNN + BiLSTM)      │    │   (ResNet-18 pretrained) │
│                          │    │                          │
│   Input: (B, 6, 200)     │    │   Input: (B, 1, 48, 48) │
│                          │    │                          │
│   Conv1D(6→32, k=7)     │    │   Modified conv1 (1-ch)  │
│   Conv1D(32→64, k=5)    │    │   Frozen layers 1-2      │
│   Conv1D(64→128, k=3)   │    │   Fine-tune layers 3-4   │
│   BiLSTM(128, h=128)     │    │   AdaptiveAvgPool2d      │
│                          │    │                          │
│   Feature: 256-dim       │    │   Feature: 256-dim       │
└──────────┬───────────────┘    └──────────┬───────────────┘
           │                               │
           └───────────┬───────────────────┘
                       ↓
             ┌─────────────────────┐
             │   FusionNet         │
             │                     │
             │   Concat (512-dim)  │
             │   FC(512→256)→BN    │
             │   FC(256→64)→ReLU   │
             │   FC(64→2)          │
             │                     │
             │   Output: Stressed  │
             │   / Not Stressed    │
             └─────────────────────┘
```

---

## Proposed Changes

### Project Structure

```
D:\Capstone\
├── config.py                  # [NEW] Hyperparams, paths, device config
├── utils.py                   # [NEW] Early stopping, metrics, plotting
├── data/
│   ├── __init__.py            # [NEW]
│   ├── physio_dataset.py      # [NEW] Physiological signal Dataset + DataLoader
│   └── facial_dataset.py      # [NEW] Facial expression Dataset + DataLoader
├── models/
│   ├── __init__.py            # [NEW]
│   ├── physio_model.py        # [NEW] 1D-CNN + BiLSTM
│   ├── facial_model.py        # [NEW] ResNet-18 transfer learning
│   └── fusion_model.py        # [NEW] Late-fusion multimodal model
├── train_physio.py            # [NEW] Train physiological model (3-class)
├── train_facial.py            # [NEW] Train facial model (2-class)
├── train_fusion.py            # [NEW] Train fused model (2-class)
├── evaluate.py                # [NEW] Full evaluation with metrics & plots
├── requirements.txt           # [NEW] Dependencies
└── checkpoints/               # [AUTO] Saved model weights
```

---

### Config and Utilities

#### [NEW] config.py
- Device auto-detection (CUDA or CPU)
- Dataset paths to all CSV files
- Hyperparameters: `lr=1e-3`, `batch_size=64` (physio) / `batch_size=8` (facial), `epochs=50`, `window_size=200`, `patience=7`
- Class weights computed from label distribution

#### [NEW] utils.py
- `EarlyStopping` class with patience and model checkpointing
- `compute_metrics()` — accuracy, precision, recall, F1 (per-class + weighted)
- `plot_training_curves()` — loss and accuracy over epochs
- `plot_confusion_matrix()` — heatmap visualization
- `plot_roc_curve()` — ROC with AUC per class

---

### Data Pipeline

#### [NEW] data/physio_dataset.py
- Loads all 3 physiological CSV files, standardizes the Subject column name
- Uses the **60-second file** as primary training data (largest: 168K samples)
- Segments by Subject to prevent data leakage between train/test
- Creates **sliding windows** of 200 timesteps with 50% overlap per subject
- Z-score normalization on 6 feature channels (eda, bvp, temp, x, y, z)
- **Subject-wise split**: Subjects 1-14 train, 15-17 val, 18-20 test
- Returns tensors: features `(6, 200)`, labels `(int)`
- Supports both 3-class and binary mode

#### [NEW] data/facial_dataset.py
- Parses 83 pixel strings into 48x48 grayscale images
- **Heavy augmentation** (training): random horizontal flip, rotation (15 degrees), affine transforms, brightness/contrast jitter, Gaussian noise
- Augmentation multiplier: generates 20x augmented copies per original image (about 1,660 training samples)
- Split: 70/15/15 stratified (58 train / 12 val / 13 test from originals)
- Normalizes to [0, 1], outputs shape `(1, 48, 48)`

---

### Models

#### [NEW] models/physio_model.py — PhysioNet
```
Conv1D(6→32, k=7) → BN → ReLU → MaxPool → Dropout(0.3)
Conv1D(32→64, k=5) → BN → ReLU → MaxPool → Dropout(0.3)
Conv1D(64→128, k=3) → BN → ReLU → MaxPool → Dropout(0.3)
BiLSTM(128, hidden=128, layers=2)
→ Last hidden concat (256-dim)
FC(256→128) → ReLU → Dropout(0.5)
FC(128→num_classes)
```
- Exposes `.get_features()` returning 256-d embedding for fusion

#### [NEW] models/facial_model.py — FacialNet
- ResNet-18 pretrained on ImageNet
- First conv modified for 1-channel grayscale input
- Freeze layers 1-2, fine-tune layers 3-4
- Head: `AdaptiveAvgPool2d → FC(512→256) → ReLU → Dropout(0.5) → FC(256→num_classes)`
- Exposes `.get_features()` returning 256-d embedding for fusion

#### [NEW] models/fusion_model.py — FusionNet
- Takes pretrained PhysioNet + FacialNet, extracts 256-d features from each
- Fusion head: `Concat(512) → FC(512→256) → BN → ReLU → Dropout(0.5) → FC(256→64) → ReLU → FC(64→2)`
- Binary classification: Stressed vs Not-Stressed

---

### Training Scripts

#### [NEW] train_physio.py
- Trains PhysioNet on 60s physiological data (3-class: Calm/Stress/Amusement)
- Weighted CrossEntropyLoss (inverse class frequency weights)
- Adam optimizer, ReduceLROnPlateau scheduler
- Early stopping (patience=7), saves best to `checkpoints/physio_best.pth`
- Prints epoch-by-epoch metrics, plots training curves

#### [NEW] train_facial.py
- Trains FacialNet on 83-image facial dataset (2-class) with 20x augmentation
- Weighted CrossEntropyLoss
- Lower learning rate for pretrained layers (1e-4) vs head (1e-3)
- Saves best to `checkpoints/facial_best.pth`

#### [NEW] train_fusion.py
- Loads pretrained physio + facial weights
- Creates paired samples (random within-class pairing)
- Maps physio labels to binary (0,2 = Not-Stressed, 1 = Stressed)
- Two-phase training: (1) frozen sub-models, train fusion head, (2) end-to-end fine-tuning
- Saves best to `checkpoints/fusion_best.pth`

---

### Evaluation

#### [NEW] evaluate.py
- Evaluates any saved model on its test set
- Reports: accuracy, precision, recall, F1 (per-class and weighted avg)
- Generates: confusion matrix PNG, ROC curve PNG, classification report
- Supports `--model physio|facial|fusion` argument

---

## Verification Plan

### Automated Tests
1. `python train_physio.py` — target over 85% val accuracy
2. `python train_facial.py` — target over 70% val accuracy (small dataset)
3. `python train_fusion.py` — expect improvement over individual models
4. `python evaluate.py --model physio` — full physiological evaluation
5. `python evaluate.py --model facial` — full facial evaluation
6. `python evaluate.py --model fusion` — full fusion evaluation

### Output Artifacts
- `checkpoints/physio_best.pth` — for later wearable inference
- `checkpoints/facial_best.pth` — for later camera inference
- `checkpoints/fusion_best.pth` — for combined real-time system
- Training curves, confusion matrices, ROC curves (PNG files)
