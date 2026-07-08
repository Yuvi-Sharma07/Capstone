"""
FacialNet: ResNet-18 Transfer Learning for Facial Expression Classification.

Uses a pretrained ResNet-18 backbone with a modified first conv layer
for single-channel (grayscale) 48x48 input. Early layers are frozen
to prevent overfitting on the small 83-image dataset.

Architecture:
    Modified ResNet-18 → AdaptiveAvgPool → FC(512→256) → FC(256→num_classes)
"""

import torch
import torch.nn as nn
from torchvision import models


class FacialNet(nn.Module):
    """
    ResNet-18 based facial expression classifier with transfer learning.

    Layers 1-2 are frozen (pretrained features), layers 3-4 are fine-tuned,
    and a custom classification head is trained from scratch.
    """

    def __init__(self, num_classes=2, dropout=0.5):
        super(FacialNet, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = 256

        # Load pretrained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first conv layer: 3-channel → 1-channel (grayscale)
        # Average the pretrained weights across the 3 input channels
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize with mean of pretrained RGB weights
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(
                original_conv1.weight.mean(dim=1, keepdim=True)
            )

        # ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # Frozen
        self.layer2 = resnet.layer2  # Frozen
        self.layer3 = resnet.layer3  # Fine-tuned
        self.layer4 = resnet.layer4  # Fine-tuned

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze early layers
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

        # Custom classifier head
        self.feature_layer = nn.Sequential(
            nn.Linear(512, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def get_features(self, x):
        """
        Extract 256-d feature embedding for fusion.

        Args:
            x: (batch, 1, 48, 48) — grayscale facial images

        Returns:
            features: (batch, 256)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, 512)
        features = self.feature_layer(x)  # (batch, 256)
        return features

    def forward(self, x):
        """
        Full forward pass.

        Args:
            x: (batch, 1, 48, 48) — grayscale facial images

        Returns:
            logits: (batch, num_classes)
        """
        features = self.get_features(x)
        logits = self.classifier(features)
        return logits

    def get_param_groups(self, pretrained_lr=1e-4, head_lr=1e-3):
        """
        Get parameter groups with differential learning rates.
        Lower LR for pretrained layers, higher LR for new head.
        """
        pretrained_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "feature_layer" in name or "classifier" in name:
                head_params.append(param)
            else:
                pretrained_params.append(param)

        return [
            {"params": pretrained_params, "lr": pretrained_lr},
            {"params": head_params, "lr": head_lr},
        ]


if __name__ == "__main__":
    # Quick test with dummy data
    model = FacialNet(num_classes=2)
    print(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}")

    # Test forward pass
    x = torch.randn(4, 1, 48, 48)  # batch=4, 1ch, 48x48
    output = model(x)
    features = model.get_features(x)
    print(f"\nInput:    {x.shape}")
    print(f"Output:   {output.shape}")     # (4, 2)
    print(f"Features: {features.shape}")   # (4, 256)
