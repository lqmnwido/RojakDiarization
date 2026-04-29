import argparse
import torchaudio
import os
import sys
import torch

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diarization.neural_pipeline import EENDPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Neural Diarization")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/diarizer_epoch_49.pt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        return

    # Initialize and run neural pipeline
    print(f"Loading model from {args.checkpoint}...")
    pipeline = EENDPipeline(model_path=args.checkpoint, n_speakers=2, device=args.device)
    
    print(f"Processing {args.audio_path}...")
    segments = pipeline.process(args.audio_path)

    # Output results
    print("\n" + "="*40)
    print(f"{'START':<10} | {'END':<10} | {'SPEAKER':<10}")
    print("-"*40)
    if not segments:
        print("No speech detected.")
    for seg in segments:
        print(f"{seg['start']:10.2f} | {seg['end']:10.2f} | {seg['speaker']}")
    print("="*40)

if __name__ == "__main__":
    main()
