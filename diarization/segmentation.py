import numpy as np
import torch

def sliding_windows(wav, sr, window=1.0, hop=0.5):
    """
    Improved granularity for short utterances.
    window: 1.0s (reduced from 1.5s)
    hop: 0.5s (reduced from 0.75s)
    """
    win_len = int(window * sr)
    hop_len = int(hop * sr)
    
    # If audio is shorter than window, process it as a single window
    if len(wav) < win_len:
        return [{
            'waveform': wav,
            'start_sample': 0,
            'end_sample': len(wav)
        }]

    segments = []
    for start in range(0, len(wav) - win_len + 1, hop_len):
        segments.append({
            'waveform': wav[start : start + win_len],
            'start_sample': start,
            'end_sample': start + win_len
        })
        
    # Capture the "tail" if it's significant (> 0.2s)
    last_end = segments[-1]['end_sample'] if segments else 0
    if len(wav) - last_end > int(0.2 * sr):
        segments.append({
            'waveform': wav[len(wav)-win_len:], # take last full window
            'start_sample': len(wav)-win_len,
            'end_sample': len(wav)
        })

    return segments
