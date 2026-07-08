"""
Physiological Signal Dataset with Survey Feature Integration.

Loads physiological signals from the WorkStress3D dataset, enriches them
with survey data (PSS, PANAS, demographics), and creates sliding windows
for training a 1D-CNN + BiLSTM model.

Subject-wise splits prevent data leakage between train/val/test.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

import config


def load_survey_features():
    """
    Load and encode survey data (PSS, PANAS, Demographics) per subject.
    Returns a dict: {subject_id: np.array of survey features}.
    """
    # 1. PSS — General Stress Test (1 feature: TotalPoints)
    pss = pd.read_csv(config.SURVEY_PSS_PATH)
    pss = pss[["Subject", "TotalPoints"]]

    # 2. PANAS — Positive and Negative Affect (10 features: C1-C10)
    panas = pd.read_csv(config.SURVEY_PANAS_PATH)
    panas_cols = [c for c in panas.columns if c.startswith("C")]

    # 3. Demographics (6 features: Gender, Age, Height, Weight, MaritalStatus, SmokingStatus)
    demo = pd.read_csv(config.SURVEY_DEMO_PATH, encoding="utf-8")
    # Encode categorical columns
    le_gender = LabelEncoder()
    demo["Gender_enc"] = le_gender.fit_transform(demo["Gender"])
    le_marital = LabelEncoder()
    demo["Marital_enc"] = le_marital.fit_transform(demo["MaritalStatus"])
    le_smoking = LabelEncoder()
    demo["Smoking_enc"] = le_smoking.fit_transform(demo["SmokingStatus"])

    demo_features = demo[["Subject", "Gender_enc", "Age", "Height(cm)", "Weight(kg)",
                           "Marital_enc", "Smoking_enc"]]

    # Merge all survey data by Subject
    survey = pss.merge(panas, on="Subject").merge(demo_features, on="Subject")

    # Normalize numeric survey features
    feature_cols = [c for c in survey.columns if c != "Subject"]
    scaler = StandardScaler()
    survey[feature_cols] = scaler.fit_transform(survey[feature_cols])

    # Build dict: subject_id → feature vector
    survey_dict = {}
    for _, row in survey.iterrows():
        subj = int(row["Subject"])
        features = row[feature_cols].values.astype(np.float32)
        survey_dict[subj] = features

    print(f"  Loaded survey features: {len(feature_cols)} features for {len(survey_dict)} subjects")
    return survey_dict, len(feature_cols)


def load_physio_data(csv_path):
    """Load a physiological signals CSV and standardize column names."""
    df = pd.read_csv(csv_path)
    # Fix the typo in 30s file ('subjet' → 'Subject')
    df.columns = [c.strip() for c in df.columns]
    if "subjet" in df.columns:
        df = df.rename(columns={"subjet": "Subject"})
    return df


def create_sliding_windows(df, window_size, stride, feature_cols):
    """
    Create sliding windows from continuous time-series data per subject.
    Returns arrays of (windows, labels, subject_ids).
    """
    windows = []
    labels = []
    subjects = []

    for subj_id, group in df.groupby("Subject"):
        data = group[feature_cols].values  # (num_timesteps, num_features)
        emotion = group["emotion"].values

        num_steps = len(data)
        for start in range(0, num_steps - window_size + 1, stride):
            end = start + window_size
            window = data[start:end]  # (window_size, num_features)
            # Use the majority label in this window
            label = int(np.bincount(emotion[start:end].astype(int)).argmax())
            windows.append(window)
            labels.append(label)
            subjects.append(subj_id)

    return np.array(windows, dtype=np.float32), np.array(labels), np.array(subjects)


class PhysioDataset(Dataset):
    """
    PyTorch Dataset for physiological signals with survey features.

    Each sample contains:
    - physio_window: (6, window_size) — 6 sensor channels over time
    - survey_features: (survey_dim,) — subject-level survey features
    - label: int — emotion class
    """

    def __init__(self, windows, labels, subject_ids, survey_dict, binary_mode=False):
        """
        Args:
            windows: np.array of shape (N, window_size, 6) — physio data
            labels: np.array of shape (N,) — emotion labels
            subject_ids: np.array of shape (N,) — subject for each window
            survey_dict: dict mapping subject_id → survey feature vector
            binary_mode: if True, map labels to binary (0,2→0, 1→1)
        """
        self.windows = windows
        self.labels = labels.copy()
        self.subject_ids = subject_ids
        self.survey_dict = survey_dict
        self.binary_mode = binary_mode

        if binary_mode:
            # Map: 0(Calm)→0, 1(Stress)→1, 2(Amusement)→0
            self.labels[self.labels == 2] = 0

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Physio window: (window_size, 6) → transpose to (6, window_size) for Conv1D
        window = torch.FloatTensor(self.windows[idx]).T  # (6, window_size)

        # Survey features for this subject
        subj_id = int(self.subject_ids[idx])
        if subj_id in self.survey_dict:
            survey = torch.FloatTensor(self.survey_dict[subj_id])
        else:
            survey = torch.zeros(config.SURVEY_FEATURE_DIM)

        label = torch.LongTensor([self.labels[idx]])[0]
        return window, survey, label


def get_physio_dataloaders(binary_mode=False):
    """
    Build train/val/test DataLoaders for physiological data.

    Uses the 60-second file (168K samples — most temporal context).
    Subject-wise split: Subjects 1-14 train, 15-17 val, 18-20 test.

    Returns:
        train_loader, val_loader, test_loader, num_survey_features
    """
    print("\n" + "=" * 60)
    print("LOADING PHYSIOLOGICAL DATA")
    print("=" * 60)

    # Load survey features
    survey_dict, num_survey_features = load_survey_features()

    # Load the 60-second physiological data (largest file)
    print("  Loading physiological_signals_60sn.csv ...")
    df = load_physio_data(config.PHYSIO_60S_PATH)
    print(f"  Loaded {len(df)} samples from {df['Subject'].nunique()} subjects")

    # Normalize physiological features (z-score per feature)
    scaler = StandardScaler()
    df[config.PHYSIO_FEATURES] = scaler.fit_transform(df[config.PHYSIO_FEATURES])

    # Create sliding windows
    print(f"  Creating sliding windows (size={config.WINDOW_SIZE}, stride={config.WINDOW_STRIDE}) ...")
    windows, labels, subjects = create_sliding_windows(
        df, config.WINDOW_SIZE, config.WINDOW_STRIDE, config.PHYSIO_FEATURES
    )
    print(f"  Total windows: {len(windows)}")

    # Subject-wise split
    train_mask = np.isin(subjects, config.TRAIN_SUBJECTS)
    val_mask = np.isin(subjects, config.VAL_SUBJECTS)
    test_mask = np.isin(subjects, config.TEST_SUBJECTS)

    print(f"  Train: {train_mask.sum()} windows ({len(config.TRAIN_SUBJECTS)} subjects)")
    print(f"  Val:   {val_mask.sum()} windows ({len(config.VAL_SUBJECTS)} subjects)")
    print(f"  Test:  {test_mask.sum()} windows ({len(config.TEST_SUBJECTS)} subjects)")

    # Print class distribution
    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        lbl = labels[mask]
        dist = np.bincount(lbl.astype(int), minlength=3)
        print(f"  {name} class dist: Calm={dist[0]}, Stress={dist[1]}, Amusement={dist[2]}")

    # Create datasets
    train_ds = PhysioDataset(windows[train_mask], labels[train_mask], subjects[train_mask],
                              survey_dict, binary_mode=binary_mode)
    val_ds = PhysioDataset(windows[val_mask], labels[val_mask], subjects[val_mask],
                            survey_dict, binary_mode=binary_mode)
    test_ds = PhysioDataset(windows[test_mask], labels[test_mask], subjects[test_mask],
                             survey_dict, binary_mode=binary_mode)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.PHYSIO_BATCH_SIZE,
                               shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.PHYSIO_BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.PHYSIO_BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, num_survey_features


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader, test_loader, n_survey = get_physio_dataloaders()
    batch = next(iter(train_loader))
    window, survey, label = batch
    print(f"\nSample batch shapes:")
    print(f"  Window: {window.shape}")    # (batch, 6, 200)
    print(f"  Survey: {survey.shape}")    # (batch, 17)
    print(f"  Labels: {label.shape}")     # (batch,)
    print(f"  Survey feature dim: {n_survey}")
