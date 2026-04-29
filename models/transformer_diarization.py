import torch
import torch.nn as nn

class DiarizationTransformer(nn.Module):
    def __init__(self, n_speakers=4, d_model=256, nhead=8, num_layers=6, input_dim=80):
        """
        n_speakers: Maximum number of speakers (slots).
        d_model: Internal dimension of the transformer.
        input_dim: Dimension of input features (e.g., Mel-filterbanks).
        """
        super().__init__()

        # CNN Frontend for local feature extraction and downsampling if needed
        self.frontend = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # Transformer Encoder for long-range context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-label classification head (one sigmoid output per speaker slot)
        self.classifier = nn.Linear(d_model, n_speakers)

    def forward(self, x):
        """
        x: (batch, features, time) e.g., (B, 80, T)
        returns: (batch, time, n_speakers) probabilities
        """
        # Feature extraction
        x = self.frontend(x) # (B, d_model, T)
        
        # Prepare for transformer (B, T, d_model)
        x = x.permute(0, 2, 1)
        
        # Temporal modeling
        x = self.transformer(x) # (B, T, d_model)
        
        # Classification
        out = self.classifier(x) # (B, T, n_speakers)
        
        return torch.sigmoid(out)
