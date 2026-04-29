import torch
import torch.nn as nn

class MultiTaskSegmentationModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, input_dim=80):
        super().__init__()
        
        # CNN Frontend to capture local temporal/spectral patterns
        self.frontend = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=5, padding=2),
            # No downsampling here to keep frame-level resolution
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Specialized heads for each task
        # 1. Speech Activity Detection (SAD)
        self.sad_head = nn.Linear(d_model, 1)
        # 2. Speaker Change Detection (SCD)
        self.scd_head = nn.Linear(d_model, 1)
        # 3. Overlap Detection (OVD)
        self.ovd_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (B, features, T)
        returns: { 'sad': (B, T, 1), 'scd': (B, T, 1), 'ovd': (B, T, 1) }
        """
        # Feature extraction
        x = self.frontend(x) # (B, d_model, T)
        x = x.permute(0, 2, 1) # (B, T, d_model)
        
        # Temporal context
        x = self.transformer(x) # (B, T, d_model)
        
        return {
            'sad': torch.sigmoid(self.sad_head(x)),
            'scd': torch.sigmoid(self.scd_head(x)),
            'ovd': torch.sigmoid(self.ovd_head(x))
        }
