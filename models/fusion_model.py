"""
FusionNet: Multimodal Late-Fusion Model for Stress Detection.

Combines feature embeddings from PhysioNet (physiological signals)
and FacialNet (facial expressions) through a fusion head for
binary stress classification (Stressed / Not Stressed).

Architecture:
    PhysioNet → 256-d ┐
                      ├→ Concat(512) → FC → BN → ReLU → FC → FC → 2 classes
    FacialNet → 256-d ┘
"""

import torch
import torch.nn as nn

from models.physio_model import PhysioNet
from models.facial_model import FacialNet


class FusionNet(nn.Module):
    """
    Multimodal late-fusion model combining physiological and facial features.

    Loads pretrained sub-models, extracts 256-d feature embeddings from each,
    concatenates them, and passes through a fusion classification head.
    """

    def __init__(self, num_physio_channels=6, num_survey_features=17,
                 physio_classes=2, facial_classes=2, fusion_classes=2,
                 dropout=0.5, freeze_submodels=True):
        super(FusionNet, self).__init__()

        self.freeze_submodels = freeze_submodels

        # Sub-models (pretrained weights loaded separately)
        self.physio_model = PhysioNet(
            num_physio_channels=num_physio_channels,
            num_survey_features=num_survey_features,
            num_classes=physio_classes,
        )
        self.facial_model = FacialNet(num_classes=facial_classes)

        # Optionally freeze sub-model weights
        if freeze_submodels:
            self._freeze_submodels()

        # Fusion head: 256 (physio) + 256 (facial) = 512
        feature_dim = 256 + 256  # Both sub-models output 256-d features
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, fusion_classes),
        )

    def _freeze_submodels(self):
        """Freeze all parameters in PhysioNet and FacialNet."""
        for param in self.physio_model.parameters():
            param.requires_grad = False
        for param in self.facial_model.parameters():
            param.requires_grad = False
        print("  Sub-models frozen (only fusion head is trainable)")

    def unfreeze_submodels(self):
        """Unfreeze sub-models for end-to-end fine-tuning."""
        for param in self.physio_model.parameters():
            param.requires_grad = True
        for param in self.facial_model.parameters():
            param.requires_grad = True
        # Keep ResNet early layers frozen
        for param in self.facial_model.layer1.parameters():
            param.requires_grad = False
        for param in self.facial_model.layer2.parameters():
            param.requires_grad = False
        self.freeze_submodels = False
        print("  Sub-models unfrozen for end-to-end fine-tuning")

    def load_pretrained_weights(self, physio_path, facial_path):
        """Load pretrained weights into sub-models."""
        physio_state = torch.load(physio_path, map_location="cpu", weights_only=True)
        self.physio_model.load_state_dict(physio_state)
        print(f"  [OK] Loaded PhysioNet weights from {physio_path}")

        facial_state = torch.load(facial_path, map_location="cpu", weights_only=True)
        self.facial_model.load_state_dict(facial_state)
        print(f"  [OK] Loaded FacialNet weights from {facial_path}")

    def forward(self, physio_input, survey_input, facial_input):
        """
        Full forward pass through fusion model.

        Args:
            physio_input: (batch, 6, window_size) — physiological signals
            survey_input: (batch, survey_dim) — survey features
            facial_input: (batch, 1, 48, 48) — facial images

        Returns:
            logits: (batch, fusion_classes)
        """
        # Extract features from sub-models (no gradient if frozen)
        if self.freeze_submodels:
            with torch.no_grad():
                physio_feat = self.physio_model.get_features(physio_input, survey_input)
                facial_feat = self.facial_model.get_features(facial_input)
        else:
            physio_feat = self.physio_model.get_features(physio_input, survey_input)
            facial_feat = self.facial_model.get_features(facial_input)

        # Concatenate features
        fused = torch.cat([physio_feat, facial_feat], dim=1)  # (batch, 512)

        # Fusion classifier
        logits = self.fusion_head(fused)
        return logits


if __name__ == "__main__":
    # Quick test with dummy data
    model = FusionNet(freeze_submodels=True)
    print(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Test forward pass
    physio = torch.randn(4, 6, 200)
    survey = torch.randn(4, 17)
    facial = torch.randn(4, 1, 48, 48)

    output = model(physio, survey, facial)
    print(f"\nInputs:  physio={physio.shape}, survey={survey.shape}, facial={facial.shape}")
    print(f"Output:  {output.shape}")  # (4, 2)
