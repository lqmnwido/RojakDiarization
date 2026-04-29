import torch
import torch.optim as optim
import json
import os
import sys
import torchaudio
import numpy as np

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_diarization import DiarizationTransformer
from diarization.loss import PITLoss

def fine_tune(json_path, audio_path, model_path, output_path, epochs=10, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on {device}...")

    # 1. Load Data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    wav_data = waveform.squeeze()
    duration = len(wav_data) / 16000

    # 2. Create Labels (Grid of Time x Speakers)
    # 50 frames per second (matching 512 hop length at 16k SR)
    fps = 16000 / 512
    n_frames = int(duration * fps)
    
    # Get unique speakers from JSON
    unique_speakers = sorted(list(set(s['speaker'] for s in data['segments'])))
    spk_to_idx = {spk: i for i, spk in enumerate(unique_speakers)}
    n_speakers = len(unique_speakers)
    
    labels = torch.zeros((n_frames, n_speakers))
    for seg in data['segments']:
        start_f = int(seg['start'] * fps)
        end_f = int(seg['end'] * fps)
        spk_idx = spk_to_idx[seg['speaker']]
        labels[start_f:end_f, spk_idx] = 1.0

    # 3. Setup Model
    model = DiarizationTransformer(n_speakers=n_speakers).to(device)
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        # Note: If n_speakers mismatch, we'd need to handle the head. 
        # For simplicity, we assume the model can handle this count.
        state_dict = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except:
            print("Weight mismatch (likely speaker count). Re-initializing head...")
            # Load everything except the classifier
            model.load_state_dict({k: v for k, v in state_dict.items() if 'classifier' not in k}, strict=False)

    model.train()
    criterion = PITLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    feature_extractor = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80).to(device)

    # 4. Training loop (Small chunks to save memory)
    chunk_len = 10 # 10 second chunks
    samples_per_chunk = chunk_len * 16000
    frames_per_chunk = int(chunk_len * fps)

    for epoch in range(epochs):
        epoch_loss = 0
        n_chunks = 0
        
        for i in range(0, len(wav_data) - samples_per_chunk, samples_per_chunk):
            optimizer.zero_grad()
            
            # Get chunk
            audio_chunk = wav_data[i : i + samples_per_chunk].unsqueeze(0).to(device)
            label_start = int((i / 16000) * fps)
            label_chunk = labels[label_start : label_start + frames_per_chunk].unsqueeze(0).to(device)
            
            # Forward
            features = torch.log1p(feature_extractor(audio_chunk))
            preds = model(features)
            
            # Align lengths
            min_t = min(preds.shape[1], label_chunk.shape[1])
            loss = criterion(preds[:, :min_t, :], label_chunk[:, :min_t, :])
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_chunks += 1
            
        print(f"Epoch {epoch} | Avg Loss: {epoch_loss/max(1, n_chunks):.4f}")

    # 5. Save
    torch.save(model.state_dict(), output_path)
    print(f"Fine-tuned model saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/checkpoints/diarizer_epoch_49.pt")
    parser.add_argument("--out", type=str, default="models/checkpoints/diarizer_finetuned.pt")
    args = parser.parse_args()
    
    fine_tune(args.json, args.audio, args.model, args.out)
