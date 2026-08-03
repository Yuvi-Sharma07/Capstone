# NeuroBioSense — Multimodal Stress Detection System

NeuroBioSense is a deep-learning-based multimodal stress detection system that fuses **physiological signals** (EDA, BVP, Temperature, Accelerometer) captured from wearable sensors with **facial expression analysis** from camera inputs. The system classifies a subject's state as **Stressed** or **Not Stressed** (binary classification) using a late-fusion neural network architecture.

---

## Project Directory Structure

```
D:\Capstone\
├── config.py                  # Central configuration (hyperparams, paths, device)
├── utils.py                   # EarlyStopping, metrics, and plotting helpers
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
│
├── data/
│   ├── __init__.py
│   ├── physio_dataset.py      # Physiological signal loader with sliding windows
│   └── facial_dataset.py      # Facial expression data loader + augmentation
│
├── models/
│   ├── __init__.py
│   ├── physio_model.py        # PhysioNet (1D-CNN + BiLSTM + Survey integration)
│   ├── facial_model.py        # FacialNet (ResNet-18 Transfer Learning)
│   └── fusion_model.py        # FusionNet (Late Fusion Model)
│
├── train_physio.py            # PhysioNet training script
├── train_facial.py            # FacialNet training script
├── train_fusion.py            # FusionNet two-phase training script
├── evaluate.py                # Unified evaluation script (metrics, confusion matrix, ROC)
│
├── checkpoints/               # Directory for saving best trained models (gitignored)
│   ├── physio_best.pth        # Trained PhysioNet weights (~2.9 MB)
│   ├── facial_best.pth        # Trained FacialNet weights (~45 MB)
│   └── fusion_best.pth        # Trained FusionNet weights (~49 MB)
│
├── plots/                     # Generated training curves & evaluation heatmaps
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
└── Stress analysis from physiological data under pressure WorkStress3D Dataset/ (gitignored)
    ├── PhysiologicalSignals/   (15s, 30s, 60s CSV files)
    ├── TheFacialExpressions/   (facial_expression.csv — 83 images as pixel strings)
    └── Survey/                 (PSS, PANAS, Demographics, Instant Questionnaires)
```

---

## System Architecture

### System Block Diagram
The following block diagram demonstrates the end-to-end data pipeline and model flow of the NeuroBioSense system:

```mermaid
graph TB
    subgraph Input_Layer["Input Layer"]
        A["Wearable Glove<br/>(EDA, BVP, Temp, Accel X/Y/Z)"]
        B["Camera<br/>(48×48 Grayscale Face)"]
        C["Survey Data<br/>(PSS, PANAS, Demographics)"]
    end

    subgraph Data_Pipeline["⚙️ Data Processing Pipeline"]
        D["Sliding Window<br/>(200 timesteps, 50% overlap)"]
        E["Z-Score Normalization<br/>(6 channels)"]
        F["Image Augmentation<br/>(20× multiplier)"]
        G["Survey Feature Encoding<br/>(17 features)"]
    end

    subgraph Model_Layer["🧠 Deep Learning Models"]
        H["PhysioNet<br/>(1D-CNN + BiLSTM)<br/>→ 256-d embedding"]
        I["FacialNet<br/>(ResNet-18 Transfer)<br/>→ 256-d embedding"]
    end

    subgraph Fusion_Layer["🔗 Fusion Layer"]
        J["Late Fusion<br/>Concat(512-d) → FC → BN → ReLU → FC"]
    end

    subgraph Output_Layer["Output"]
        K["Binary Classification<br/>Not Stressed / Stressed"]
    end

    A --> D --> E --> H
    C --> G --> H
    B --> F --> I
    H --> J
    I --> J
    J --> K
```

### Component Flow Diagram
Shows how different modules and source files interact across the project:

```mermaid
graph LR
    subgraph Dataset["WorkStress3D Dataset"]
        D1["Physiological CSVs<br/>(15s, 30s, 60s)"]
        D2["Facial Expression CSV<br/>(83 images)"]
        D3["Survey CSVs<br/>(PSS, PANAS, Demo)"]
    end

    subgraph Processing["Data Processing"]
        P1["physio_dataset.py"]
        P2["facial_dataset.py"]
    end

    subgraph Models["DL Models"]
        M1["physio_model.py"]
        M2["facial_model.py"]
        M3["fusion_model.py"]
    end

    subgraph Training["Training Scripts"]
        T1["train_physio.py"]
        T2["train_facial.py"]
        T3["train_fusion.py"]
    end

    subgraph Eval["Evaluation"]
        E1["evaluate.py"]
        E2["utils.py"]
    end

    subgraph Config["Configuration"]
        C1["config.py"]
    end

    D1 --> P1 --> T1 --> M1
    D2 --> P2 --> T2 --> M2
    D3 --> P1
    M1 --> T3 --> M3
    M2 --> T3
    M1 --> E1
    M2 --> E1
    M3 --> E1
    E2 --> E1
    C1 --> T1
    C1 --> T2
    C1 --> T3
    C1 --> E1
```

---

## Technology Stack & Dependencies

The project is built using:
- **Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch $\ge$ 2.0.0
- **Computer Vision**: torchvision $\ge$ 0.15.0 (for ResNet-18 pretrained backbone)
- **Data Engineering**: Pandas, NumPy $\ge$ 1.24.0, OpenCV $\ge$ 4.8.0
- **Machine Learning Utilities**: scikit-learn $\ge$ 1.3.0 (Z-score scaling, label encoders)
- **Visualization**: Matplotlib $\ge$ 3.7.0, Seaborn $\ge$ 0.12.0

---

## Deep Learning Models

### 1. PhysioNet (`models/physio_model.py`)
- **Inputs**: Physiological signals (6 channels: EDA, BVP, Temperature, Accelerometer X/Y/Z) and Subject Survey features (17 features: PSS, PANAS, Demographics).
- **Architecture**: 
  - Three 1D convolutional layers with batch normalization, ReLU activation, max pooling, and dropout (0.3).
  - A Bidirectional LSTM (BiLSTM) with 2 layers and 128 hidden units (outputting a 256-dimensional temporal embedding).
  - Concatenation of the 256-d temporal embedding with the 17-d survey features.
  - Fully connected layers classifying into 3 classes: **Calm / Stress / Amusement**.
- **Inference**: Provides a `get_features()` method to extract the raw 256-d signal representation for downstream late fusion.

### 2. FacialNet (`models/facial_model.py`)
- **Inputs**: 48x48 pixel grayscale face images parsed from the dataset.
- **Architecture**:
  - ResNet-18 initialized with pretrained ImageNet weights. The input convolution layer is adapted from 3 channels to 1 channel (averaging weights across channels).
  - Layers 1 and 2 are frozen to leverage general low-level visual features.
  - Layers 3 and 4 are fine-tuned to capture facial expression details specific to stress.
  - A custom classifier head maps features to 2 output classes: **Non-Stress / Stress**.
- **Inference**: Provides a `get_features()` method to extract the raw 256-d spatial embedding.

### 3. FusionNet (`models/fusion_model.py`)
- **Strategy**: Late fusion matching by concatenating representations.
- **Architecture**:
  - Concatenates the 256-d temporal embedding from PhysioNet and 256-d spatial embedding from FacialNet to form a 512-d feature vector.
  - A multi-layer feed-forward neural network with batch normalization, ReLU, and dropout (0.5) outputs predictions for 2 classes: **Not Stressed / Stressed**.

---

## Training & Evaluation Pipelines

### Training Workflow Activity Diagram

```mermaid
graph TD
    Start([Start]) --> A["Load config.py"]
    A --> B{"GPU Available?"}
    B -->|Yes| C["Set device = CUDA"]
    B -->|No| D["Set device = CPU"]
    C --> E["Set random seed (42)"]
    D --> E

    E --> F["Load & preprocess<br/>physiological data"]
    F --> G["Create sliding windows<br/>(200 steps, 50% overlap)"]
    G --> H["Subject-wise split<br/>(14 / 3 / 3)"]
    H --> I["Train PhysioNet<br/>(1D-CNN + BiLSTM)"]
    I --> J{"Val loss<br/>improving?"}
    J -->|Yes| I
    J -->|No / Patience=7| K["Save physio_best.pth"]

    E --> L["Load & parse<br/>facial expression CSV"]
    L --> M["Apply 20× augmentation"]
    M --> N["Stratified split<br/>(70/15/15)"]
    N --> O["Train FacialNet<br/>(ResNet-18)"]
    O --> P{"Val loss<br/>improving?"}
    P -->|Yes| O
    P -->|No / Patience=7| Q["Save facial_best.pth"]

    K --> R["Load pretrained<br/>sub-models"]
    Q --> R
    R --> S["Create within-class<br/>paired samples"]
    S --> T["Phase 1: Frozen<br/>sub-models (25 ep)"]
    T --> U["Phase 2: End-to-end<br/>fine-tuning (25 ep)"]
    U --> V["Save fusion_best.pth"]
    V --> W["Run evaluate.py<br/>on all 3 models"]
    W --> X["Generate plots<br/>(confusion matrix, ROC)"]
    X --> End([End])
```

### FusionNet Two-Phase Training Strategy
Because the sub-models learn features from different domains, FusionNet is trained in two distinct phases:
1. **Phase 1 (Frozen Sub-models)**: The sub-model weights are frozen. Only the fusion classification head is trained for 25 epochs (Learning Rate = 1e-3). This stabilizes the joint classification boundary.
2. **Phase 2 (End-to-End Fine-Tuning)**: The sub-models are unfrozen (with early ResNet-18 layers remaining frozen) and optimized end-to-end at a lower learning rate (1e-4) for another 25 epochs.

---

## How to Run the Project

### 1. Installation
Install all required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Training the Pipeline
You must train the sub-models sequentially so that FusionNet can load their pre-trained weights.

1. **Train PhysioNet**:
   ```bash
   python train_physio.py
   ```
2. **Train FacialNet**:
   ```bash
   python train_facial.py
   ```
3. **Train FusionNet**:
   ```bash
   python train_fusion.py
   ```

### 3. Unified Evaluation
Evaluate the accuracy, precision, recall, and F1 score of any of your models using the `evaluate.py` script. The evaluation also automatically exports confusion matrix and ROC curves to the `plots/` directory:

```bash
# Evaluate Physiological Model (3-class)
python evaluate.py --model physio

# Evaluate Facial Model (2-class)
python evaluate.py --model facial

# Evaluate late Fusion Model (2-class)
python evaluate.py --model fusion
```

---

## Key Design Decisions
1. **Subject-wise Splitting**: Prevents data leakage by ensuring that data from subjects used during training are never present in the validation or test sets (Subject 1-14: Train, 15-17: Validation, 18-20: Test).
2. **Class Imbalance Handling**: Inverse-frequency weights are dynamically calculated and applied to the loss functions of all three networks.
3. **Data Augmentation**: Grayscale face datasets are augmented 20-fold using horizontal flips, random rotations, affine translations, and scaling to resolve the constraints of the small image subset (83 samples).
4. **Early Stopping**: All training runs utilize an `EarlyStopping` utility tracking validation loss (patience=7) to prevent overfitting.
