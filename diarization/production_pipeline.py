import torch
import numpy as np
import torchaudio
from .vad import VAD
from .embedding import SpeakerEmbedding
from .segmentation import sliding_windows
from .clustering import SpeakerClustering

class ProductionPipeline:
    def __init__(self, threshold=0.8, device="cpu"):
        # Pillar A: SAD (Speech Activity Detection)
        self.vad = VAD(threshold=0.35) 
        
        # Pillar B: Speaker Representation (ECAPA-TDNN)
        self.embedder = SpeakerEmbedding(device=device)
        
        # Pillar C: Clustering Engine
        self.threshold = threshold
        self.device = device
        
        # Post-processing settings
        self.min_duration_on = 0.2
        self.min_duration_off = 0.5

    def process(self, audio_path, n_speakers=None):
        """
        Production entry point with multi-stage processing.
        """
        # 1. Preprocessing (Normalization & Loading)
        waveform, sr = self._load_and_normalize(audio_path)
        wav_data = waveform.squeeze()

        # 2. Stage 1: Segmentation (VAD/SAD)
        # Identifies WHERE speech is happening.
        speech_segments = self.vad.get_speech_segments(wav_data, sr)
        if not speech_segments:
            return []

        # 3. Stage 2: Feature Extraction (Embeddings)
        # Converts audio chunks into 192D identity vectors.
        embeddings = []
        metadata = []
        for seg in speech_segments:
            seg_wav = wav_data[seg["start"]:seg["end"]]
            # Higher resolution windows for better accuracy
            windows = sliding_windows(seg_wav, sr, window=1.0, hop=0.5)
            for win in windows:
                emb = self.embedder.extract(win['waveform'])
                embeddings.append(emb)
                metadata.append({
                    'start': (seg['start'] + win['start_sample']) / sr,
                    'end': (seg['start'] + win['end_sample']) / sr
                })

        if not embeddings:
            return []

        # 4. Stage 3: Speaker Identity Assignment (Clustering)
        clusterer = SpeakerClustering(threshold=self.threshold)
        if n_speakers:
            from sklearn.cluster import AgglomerativeClustering
            clusterer.clusterer = AgglomerativeClustering(
                n_clusters=n_speakers, metric="precomputed", linkage="average"
            )

        embeddings_stack = np.stack(embeddings)
        labels = clusterer.cluster(embeddings_stack)

        # 5. Stage 4: Inference Refinement
        from .utils import smooth_predictions, compute_clustering_confidence
        
        # A. Median Smoothing (removes rapid, unrealistic speaker switching)
        labels = smooth_predictions(list(labels), window_size=5)
        
        # B. Confidence Calculation
        confidences = compute_clustering_confidence(embeddings_stack, np.array(labels))

        # 6. Stage 5: Post-processing & Timeline Refinement
        raw_results = []
        for i, label in enumerate(labels):
            # C. Confidence Thresholding: Ignore low-confidence segments
            if confidences[i] < 0.6:
                continue
                
            raw_results.append({
                'speaker': f"SPEAKER_{label:02d}",
                'start': metadata[i]['start'],
                'end': metadata[i]['end'],
                'confidence': round(float(confidences[i]), 2)
            })
            
        return self._refine_timeline(raw_results)

    def _load_and_normalize(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, 16000

    def _refine_timeline(self, segments):
        """Production-grade smoothing and merging."""
        if not segments: return []
        
        # Sort by time
        segments = sorted(segments, key=lambda x: x['start'])
        
        merged = []
        current = segments[0].copy()
        
        for nxt in segments[1:]:
            # If same speaker and gap is smaller than min_duration_off
            if nxt['speaker'] == current['speaker'] and nxt['start'] <= current['end'] + self.min_duration_off:
                current['end'] = max(current['end'], nxt['end'])
            else:
                # Only keep segments longer than min_duration_on (filter noise)
                if (current['end'] - current['start']) >= self.min_duration_on:
                    merged.append(current)
                current = nxt.copy()
        
        merged.append(current)
        return merged
