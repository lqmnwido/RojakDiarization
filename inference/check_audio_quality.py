import torchaudio
import torch
import numpy as np
import os

def check_audio(file_path):
    if not os.path.exists(file_path):
        return {"error": f"File {file_path} not found."}

    try:
        # Load audio
        waveform, sr = torchaudio.load(file_path)
        
        # Basic properties
        duration = waveform.shape[1] / sr
        channels = waveform.shape[0]
        
        # Convert to mono for analysis
        mono_wav = torch.mean(waveform, dim=0) if channels > 1 else waveform.squeeze()
        
        # 1. Volume Analysis (RMS)
        rms = torch.sqrt(torch.mean(mono_wav**2)).item()
        peak = torch.max(torch.abs(mono_wav)).item()
        
        # 2. Silence/Speech Ratio
        # Simple energy-based VAD for diagnostic
        frame_len = int(0.02 * sr) # 20ms frames
        frames = mono_wav.unfold(0, frame_len, frame_len)
        energies = torch.sqrt(torch.mean(frames**2, dim=1))
        
        # Threshold for "audible" content
        audible_threshold = 0.01 
        audible_frames = (energies > audible_threshold).float().mean().item()
        
        # 3. Dynamic Range
        # If RMS is very low (e.g., < 0.001), it's basically silence or very faint
        status = "Good"
        if rms < 0.005:
            status = "Too Quiet/Silent"
        elif peak > 0.99:
            status = "Clipping/Distorted"
        elif audible_frames < 0.05:
            status = "Mostly Silence"

        return {
            "status": status,
            "technical": {
                "sample_rate": sr,
                "channels": channels,
                "duration_sec": round(duration, 2),
                "file_size_mb": round(os.path.getsize(file_path) / (1024*1024), 2)
            },
            "audio_quality": {
                "rms_volume": round(rms, 4),
                "peak_amplitude": round(peak, 4),
                "audible_speech_estimate": f"{audible_frames*100:.1f}%"
            }
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import sys
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else "output_audio_1.wav"
    results = check_audio(path)
    print(json.dumps(results, indent=2))
