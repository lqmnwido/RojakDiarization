import os
import argparse
from dotenv import load_dotenv
from src.diarization.pipeline import DiarizationPipeline

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Open-source Speaker Diarization Pipeline")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers (optional)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Clustering threshold (default: 0.7)")
    
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: File {args.audio_path} not found.")
        return

    print("Initializing pipeline...")
    pipeline = DiarizationPipeline(
        n_speakers=args.num_speakers, 
        distance_threshold=args.threshold
    )

    print(f"Processing {args.audio_path}...")
    segments = pipeline.process(args.audio_path)

    print("\nDiarization Results:")
    print("-" * 30)
    for seg in segments:
        print(f"[{seg['start']:7.2f}s - {seg['end']:7.2f}s] {seg['speaker']}")
    print("-" * 30)

if __name__ == "__main__":
    main()
