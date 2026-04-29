import torch
import numpy as np
import torchaudio
from models.transformer_diarization import DiarizationTransformer

class EENDPipeline:
    def __init__(self, model_path=None, n_speakers=4, device="cpu"):
        self.device = device
        self.n_speakers = n_speakers
        self.model = DiarizationTransformer(n_speakers=n_speakers).to(device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Feature extractor
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80
        ).to(device)

    def process(self, audio_path, threshold=0.5, chunk_len_sec=10.0, step_len_sec=5.0):
        """
        Runs neural diarization in chunks to avoid CUDA Out of Memory.
        """
        # Load and preprocess
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        # Normalize audio power (Ensures quiet speakers are heard)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        
        # Audio info
        total_samples = waveform.shape[1]
        chunk_samples = int(chunk_len_sec * 16000)
        step_samples = int(step_len_sec * 16000)
        
        # We will store probabilities for each frame
        # MelSpectrogram default hop_length is 512
        hop_length = 512
        total_frames = (total_samples // hop_length) + 1
        combined_probs = np.zeros((total_frames, self.n_speakers))
        count_map = np.zeros((total_frames, 1))

        # Sliding Window Processing
        for start in range(0, total_samples, step_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end].to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(chunk) # (1, 80, T_chunk)
                if features.dim() == 2: features = features.unsqueeze(0)
                
                preds = self.model(features) # (1, T_chunk, n_speakers)
                chunk_probs = preds[0].cpu().numpy()
            
            # Place chunk probs into the global timeline
            start_frame = start // hop_length
            end_frame = start_frame + chunk_probs.shape[0]
            
            # Handle possible frame mismatch at the very end
            actual_end_frame = min(end_frame, total_frames)
            chunk_slice = chunk_probs[:(actual_end_frame - start_frame)]
            
            combined_probs[start_frame:actual_end_frame] += chunk_slice
            count_map[start_frame:actual_end_frame] += 1
            
            if end == total_samples: break

        # Average the overlapping areas
        combined_probs = np.divide(combined_probs, count_map, out=np.zeros_like(combined_probs), where=count_map!=0)
        
        # Convert to segments
        frame_duration = hop_length / 16000 
        results = []
        for s in range(self.n_speakers):
            active_frames = combined_probs[:, s] > threshold
            start_frame = None
            for f in range(len(active_frames)):
                is_active = bool(active_frames[f])
                if is_active and start_frame is None:
                    start_frame = f
                elif not is_active and start_frame is not None:
                    results.append({
                        'speaker': f"SPEAKER_{s:02d}",
                        'start': start_frame * frame_duration,
                        'end': f * frame_duration
                    })
                    start_frame = None
            
            if start_frame is not None:
                results.append({
                    'speaker': f"SPEAKER_{s:02d}",
                    'start': start_frame * frame_duration,
                    'end': len(active_frames) * frame_duration
                })

        return sorted(results, key=lambda x: x['start'])
