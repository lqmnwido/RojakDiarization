import torch
import torchaudio
import numpy as np

def load_audio(path, sample_rate=16000):
    """Load audio and resample if necessary."""
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform.squeeze(), sample_rate

def segment_audio(waveform, sample_rate, window_size_s=2.0, hop_size_s=1.0):
    """Split waveform into overlapping segments."""
    window_size = int(window_size_s * sample_rate)
    hop_size = int(hop_size_s * sample_rate)
    
    segments = []
    start = 0
    while start + window_size <= len(waveform):
        segments.append({
            'waveform': waveform[start:start + window_size],
            'start': start / sample_rate,
            'end': (start + window_size) / sample_rate
        })
        start += hop_size
        
    return segments

def smooth_predictions(labels, window_size=3):
    """
    Applies median filtering to label sequences to remove jitter.
    labels: List of speaker IDs.
    """
    if len(labels) < window_size:
        return labels
    
    smoothed = []
    for i in range(len(labels)):
        start = max(0, i - window_size // 2)
        end = min(len(labels), i + window_size // 2 + 1)
        window = labels[start:end]
        # Most frequent label in the window
        smoothed.append(max(set(window), key=window.count))
    return smoothed

def compute_clustering_confidence(embeddings, labels):
    """
    Computes a simple confidence score based on distance to cluster centroid.
    """
    confidences = []
    unique_labels = np.unique(labels)
    centroids = {l: np.mean(embeddings[labels == l], axis=0) for l in unique_labels}
    
    for i, emb in enumerate(embeddings):
        label = labels[i]
        centroid = centroids[label]
        # Cosine similarity as confidence
        sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
        confidences.append(sim)
        
    return confidences
