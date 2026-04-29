import os
import sys
import argparse
import pandas as pd
import torch
import json

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from diarization.hybrid_pipeline import HybridPipeline

def segments_to_annotation(segments, uri="audio"):
    """Converts pipeline output to pyannote Annotation object."""
    annotation = Annotation(uri=uri)
    for seg in segments:
        annotation[Segment(seg['start'], seg['end'])] = str(seg['speaker'])
    return annotation

def load_reference(ref_path):
    """Loads reference from CSV or JSON."""
    if ref_path.endswith('.json'):
        with open(ref_path, 'r') as f:
            data = json.load(f)
        return data['segments']
    elif ref_path.endswith('.csv'):
        df = pd.read_csv(ref_path)
        return df.to_dict('records')
    else:
        raise ValueError("Reference must be .csv or .json")

def main():
    parser = argparse.ArgumentParser(description="Diarization Benchmarking")
    parser.add_argument("--audio", type=str, required=True, help="Path to test audio")
    parser.add_argument("--ref", type=str, required=True, help="Path to ground truth (CSV or JSON)")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/diarizer_finetuned.pt")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Detect speaker count from checkpoint to avoid size mismatch
    n_speakers = None
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'classifier.weight' in ckpt:
            n_speakers = ckpt['classifier.weight'].shape[0]
            print(f"Detected {n_speakers} speakers in checkpoint.")

    # Use HybridPipeline to match our production logic
    pipeline = HybridPipeline(model_path=args.checkpoint, n_speakers=n_speakers, device=device)
    
    print(f"Loading reference from {args.ref}...")
    ref_segments = load_reference(args.ref)
    ref_ann = segments_to_annotation(ref_segments, uri="reference")
    
    print(f"Running inference with {args.checkpoint}...")
    hyp_segments = pipeline.process(args.audio)
    hyp_ann = segments_to_annotation(hyp_segments, uri="hypothesis")
    
    # Compute DER
    metric = DiarizationErrorRate()
    der_results = metric(ref_ann, hyp_ann, detailed=True)
    
    print("\n" + "="*40)
    print(f"BENCHMARK RESULTS: {os.path.basename(args.audio)}")
    print("="*40)
    print(f"{'DER (Error Rate)':<20}: {der_results['diarization error rate']*100:>6.2f}%")
    print(f"{'Confusion':<20}: {der_results['confusion']*100:>6.2f}%")
    print(f"{'Missed Detection':<20}: {der_results['missed detection']*100:>6.2f}%")
    print(f"{'False Alarm':<20}: {der_results['false alarm']*100:>6.2f}%")
    print("-" * 40)
    print(f"{'Total Speech':<20}: {der_results['total']:>7.2f}s")
    print("="*40)
    print("\nNOTE: Lower DER is better. 0% is perfect.")

if __name__ == "__main__":
    main()
