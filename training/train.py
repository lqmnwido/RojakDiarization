import torch
import torch.optim as optim
import os
import sys
from torch.utils.data import DataLoader

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_diarization import DiarizationTransformer
from diarization.loss import PITLoss
from training.dataset import ProductionDiarizationDataset, diarization_collate_fn

def build_speaker_map(data_dir):
    """Scan the speakers directory and map files to speaker IDs."""
    speaker_map = {}
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found. Run download_data.py first.")
        
    for spk_id in os.listdir(data_dir):
        path = os.path.join(data_dir, spk_id)
        if os.path.isdir(path):
            # Find all flac files (LibriSpeech format)
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.flac', '.wav'))]
            if files:
                speaker_map[spk_id] = files
    
    print(f"Loaded {len(speaker_map)} speakers for training.")
    return speaker_map

def train_neural_diarizer(speaker_data, epochs=50, batch_size=4, lr=1e-4):
    # Check for GPU (highly recommended for training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training started on: {device}")

    # 1. Initialize Dataset & Loader
    # Using 2 speakers per mix and 10-second chunks
    dataset = ProductionDiarizationDataset(speaker_data, n_speakers=2, segment_len=10)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=diarization_collate_fn,
        num_workers=0 # Set to 0 if on Windows to avoid pickling errors
    )

    # 2. Initialize Model (4 slots for speakers), Loss, Optimizer
    # d_model=256, nhead=8 matches the architecture we built
    model = DiarizationTransformer(n_speakers=2, d_model=256).to(device)
    criterion = PITLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ensure output directory exists
    os.makedirs("models/checkpoints", exist_ok=True)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (features, labels) in enumerate(dataloader):
            # features: (B, 1, 80, T_feat) -> needs to be (B, 80, T)
            # Torchaudio MelSpec adds a channel dim [1]
            features = features.squeeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            preds = model(features)
            
            # Since features and labels might have slight temporal mismatch 
            # due to windowing, we trim them to match the smallest length
            min_time = min(preds.shape[1], labels.shape[1])
            preds = preds[:, :min_time, :]
            labels = labels[:, :min_time, :]
            
            # Compute PIT Loss
            loss = criterion(preds, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"==> Epoch {epoch} Complete | Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"models/checkpoints/diarizer_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved: {checkpoint_path}")

if __name__ == "__main__":
    try:
        spk_map = build_speaker_map("data/speakers")
        train_neural_diarizer(spk_map)
    except Exception as e:
        print(f"Error: {e}")
