"""
PhysioNet: 1D-CNN + BiLSTM for Physiological Signal Classification.

Processes 6-channel time-series sensor data (EDA, BVP, Temperature,
Accelerometer X/Y/Z) with survey features (PSS, PANAS, Demographics)
for stress detection.

Architecture:
    Conv1D ×3 → BiLSTM → concat with survey features → FC layers → output
"""

import torch
import torch.nn as nn


class PhysioNet(nn.Module):
    """
    1D-CNN + Bidirectional LSTM model for physiological time-series classification.

    The CNN extracts local temporal patterns, the BiLSTM captures long-range
    dependencies, and survey features provide subject-level context.
    """

    def __init__(self, num_physio_channels=6, num_survey_features=17,
                 num_classes=3, dropout=0.5):
        super(PhysioNet, self).__init__()

        self.num_survey_features = num_survey_features
        self.num_classes = num_classes

        # ── 1D Convolutional Blocks ──
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(num_physio_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )

        # ── Bidirectional LSTM ──
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # ── Feature dimension: BiLSTM(256) + survey features ──
        self.feature_dim = 256  # BiLSTM last hidden (128*2)
        combined_dim = self.feature_dim + num_survey_features

        # ── Classifier Head ──
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, physio_input, survey_input):
        """
        Extract feature embeddings from physiological and survey data.

        Args:
            physio_input: (batch, 6, window_size) — sensor channels over time
            survey_input: (batch, survey_dim) — subject-level survey features

        Returns:
            features: (batch, 256 + survey_dim) — combined feature vector
        """
        # CNN: (batch, 6, window) → (batch, 128, window//8)
        x = self.conv_block1(physio_input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Reshape for LSTM: (batch, channels, time) → (batch, time, channels)
        x = x.permute(0, 2, 1)

        # BiLSTM: take the last hidden state from both directions
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers*2, batch, hidden_size)
        # Concatenate last forward and backward hidden states
        h_forward = h_n[-2]  # Last forward layer
        h_backward = h_n[-1]  # Last backward layer
        lstm_features = torch.cat([h_forward, h_backward], dim=1)  # (batch, 256)

        # Concatenate with survey features
        combined = torch.cat([lstm_features, survey_input], dim=1)
        return combined

    def get_features(self, physio_input, survey_input):
        """
        Get the 256-d physiological feature embedding (without survey)
        for use in the fusion model.
        """
        x = self.conv_block1(physio_input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        return torch.cat([h_forward, h_backward], dim=1)  # (batch, 256)

    def forward(self, physio_input, survey_input):
        """
        Full forward pass: features → classifier → logits.

        Args:
            physio_input: (batch, 6, window_size)
            survey_input: (batch, survey_dim)

        Returns:
            logits: (batch, num_classes)
        """
        features = self.extract_features(physio_input, survey_input)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    # Quick test with dummy data
    model = PhysioNet(num_physio_channels=6, num_survey_features=17, num_classes=3)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    physio = torch.randn(4, 6, 200)   # batch=4, 6 channels, 200 timesteps
    survey = torch.randn(4, 17)        # batch=4, 17 survey features
    output = model(physio, survey)
    print(f"Input:  physio={physio.shape}, survey={survey.shape}")
    print(f"Output: {output.shape}")    # (4, 3)
