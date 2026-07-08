"""
Facial Expression Dataset with Heavy Augmentation.

Loads 83 facial expression images from the WorkStress3D dataset,
applies 20x augmentation to compensate for the tiny dataset size,
and prepares data for training a ResNet-18 transfer learning model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config


def parse_pixels(pixel_string):
    """Convert pixel string '105 106 105 ...' to 48x48 numpy array."""
    pixels = np.array(pixel_string.split(), dtype=np.float32)
    image = pixels.reshape(config.FACIAL_IMAGE_SIZE, config.FACIAL_IMAGE_SIZE)
    return image


class FacialDataset(Dataset):
    """
    PyTorch Dataset for facial expression classification.

    Supports heavy data augmentation for training on the small 83-image dataset.
    Each image is 48x48 grayscale, output as (1, 48, 48) tensor.
    """

    def __init__(self, images, labels, augment=False, aug_multiplier=1):
        """
        Args:
            images: np.array of shape (N, 48, 48) — grayscale images
            labels: np.array of shape (N,) — emotion labels (0 or 1)
            augment: whether to apply data augmentation
            aug_multiplier: how many augmented copies per original image
        """
        self.original_images = images
        self.original_labels = labels
        self.augment = augment
        self.aug_multiplier = aug_multiplier if augment else 1

        # Define augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1),
                scale=(0.9, 1.1), shear=10
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),  # Converts to (1, H, W) and scales to [0, 1]
        ])

        self.basic_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.original_images) * self.aug_multiplier

    def __getitem__(self, idx):
        # Map augmented index back to original image
        orig_idx = idx % len(self.original_images)
        image = self.original_images[orig_idx].copy()  # (48, 48)
        label = self.original_labels[orig_idx]

        # Convert to uint8 for PIL transforms
        image_uint8 = image.astype(np.uint8)

        if self.augment and idx >= len(self.original_images):
            # Apply augmentation for augmented copies
            image_tensor = self.augment_transform(image_uint8)
        else:
            # Basic transform for original or validation images
            image_tensor = self.basic_transform(image_uint8)

        label = torch.LongTensor([label])[0]
        return image_tensor, label


def get_facial_dataloaders():
    """
    Build train/val/test DataLoaders for facial expression data.

    Loads 83 images, splits 70/15/15 (stratified), and applies
    20x augmentation on training set.

    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "=" * 60)
    print("LOADING FACIAL EXPRESSION DATA")
    print("=" * 60)

    # Load CSV
    df = pd.read_csv(config.FACIAL_PATH)
    print(f"  Loaded {len(df)} facial images")
    print(f"  Class distribution: {df['emotion'].value_counts().to_dict()}")

    # Parse pixel strings to images
    images = np.array([parse_pixels(p) for p in df["pixels"].values])  # (83, 48, 48)
    labels = df["emotion"].values.astype(np.int64)

    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.30, random_state=config.RANDOM_SEED,
        stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=config.RANDOM_SEED,
        stratify=y_temp
    )

    print(f"  Train: {len(X_train)} images (before augmentation)")
    print(f"  Val:   {len(X_val)} images")
    print(f"  Test:  {len(X_test)} images")

    # Create datasets
    train_ds = FacialDataset(
        X_train, y_train,
        augment=True, aug_multiplier=config.FACIAL_AUG_MULTIPLIER
    )
    val_ds = FacialDataset(X_val, y_val, augment=False)
    test_ds = FacialDataset(X_test, y_test, augment=False)

    print(f"  Train dataset size (with {config.FACIAL_AUG_MULTIPLIER}x aug): {len(train_ds)}")

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.FACIAL_BATCH_SIZE,
                               shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.FACIAL_BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.FACIAL_BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader, test_loader = get_facial_dataloaders()
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"\nSample batch shapes:")
    print(f"  Images: {images.shape}")   # (batch, 1, 48, 48)
    print(f"  Labels: {labels.shape}")   # (batch,)
    print(f"  Pixel range: [{images.min():.3f}, {images.max():.3f}]")
