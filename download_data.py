import os
import shutil
import torchaudio
from pathlib import Path

def download_and_setup():
    # 1. Define paths
    raw_data_path = Path("data/raw_librispeech")
    processed_path = Path("data/speakers")
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)

    print("--- Starting LibriSpeech Download (dev-clean subset, ~330MB) ---")
    # This downloads and extracts automatically
    torchaudio.datasets.LIBRISPEECH(root=raw_data_path, url="dev-clean", download=True)
    print("Download complete.")

    print("--- Reorganizing files into speaker folders ---")
    # Path where torchaudio extracts: data/raw_librispeech/LibriSpeech/dev-clean/SPEAKER_ID/CHAPTER_ID/FILE.flac
    base_dir = raw_data_path / "LibriSpeech" / "dev-clean"
    
    count = 0
    for speaker_id in os.listdir(base_dir):
        speaker_path = base_dir / speaker_id
        if not speaker_path.is_dir():
            continue
            
        target_spk_dir = processed_path / f"spk_{speaker_id}"
        target_spk_dir.mkdir(exist_ok=True)
        
        # Walk through all chapters for this speaker
        for root, _, files in os.walk(speaker_path):
            for f in files:
                if f.endswith(".flac"):
                    # Move and rename slightly to avoid collisions
                    src = Path(root) / f
                    dst = target_spk_dir / f
                    shutil.move(str(src), str(dst))
                    count += 1
                    
    print(f"Success! Organized {count} utterances across {len(os.listdir(processed_path))} speakers.")
    print(f"Your training data is ready in: {processed_path}")

if __name__ == "__main__":
    download_and_setup()
