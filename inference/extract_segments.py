import torchaudio
import json
import os
import argparse

def extract_segments(audio_path, json_path, output_dir="data/clips", max_clips_per_speaker=3):
    """
    Extracts small audio clips for each speaker identified in the JSON.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio {audio_path} not found.")
        return

    # 1. Load Data
    waveform, sr = torchaudio.load(audio_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Group segments by speaker
    speaker_segments = {}
    for seg in data['segments']:
        spk = seg['speaker']
        if spk not in speaker_segments:
            speaker_segments[spk] = []
        speaker_segments[spk].append(seg)

    print(f"Extracting sample clips to: {output_dir}")
    print("-" * 30)

    # 3. Extract ALL clips
    for spk, segments in speaker_segments.items():
        for i, seg in enumerate(segments):
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            
            # Extract and save
            clip = waveform[:, start_sample:end_sample]
            
            # Use padded index to keep files sorted
            filename = f"{spk}_clip_{i+1:03d}_{seg['start']}s.wav"
            out_path = os.path.join(output_dir, filename)
            torchaudio.save(out_path, clip, sr)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: Extracted {i+1} clips for {spk}...")
        
        print(f"Finished {spk}: {len(segments)} clips extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/clips")
    args = parser.parse_args()
    
    extract_segments(args.audio, args.json, args.out)
