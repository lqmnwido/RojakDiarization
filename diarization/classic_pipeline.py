import torch
import numpy as np
import torchaudio
from .vad import VAD
from .embedding import SpeakerEmbedding
from .segmentation import sliding_windows
from .clustering import SpeakerClustering

class ClassicPipeline:
    def __init__(self, threshold=0.8, device="cpu"):
        # Higher default threshold (0.8) helps prevent over-segmentation
        self.vad = VAD(threshold=0.35)
        self.embedder = SpeakerEmbedding(device=device)
        self.threshold = threshold
        self.device = device

    def process(self, audio_path, n_speakers=None):
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        # ... (resampling logic same)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        wav = waveform.squeeze()

        # 1. VAD
        speech_segments = self.vad.get_speech_segments(wav, sr)
        if not speech_segments: return []

        # 2. Embeddings
        all_embeddings = []
        metadata = []
        for seg in speech_segments:
            seg_wav = wav[seg["start"]:seg["end"]]
            windows = sliding_windows(seg_wav, sr)
            for win in windows:
                emb = self.embedder.extract(win['waveform'])
                all_embeddings.append(emb)
                metadata.append({
                    'start': (seg['start'] + win['start_sample']) / sr,
                    'end': (seg['start'] + win['end_sample']) / sr
                })

        if not all_embeddings: return []

        # 3. Clustering (Dynamic threshold/n_speakers)
        clusterer = SpeakerClustering(threshold=self.threshold)
        if n_speakers:
            # If we know the speaker count, we force the clusterer
            from sklearn.cluster import AgglomerativeClustering
            clusterer.clusterer = AgglomerativeClustering(
                n_clusters=n_speakers,
                metric="precomputed",
                linkage="average"
            )

        labels = clusterer.cluster(np.stack(all_embeddings))

        # 4. Merge
        results = []
        for i, label in enumerate(labels):
            results.append({
                'speaker': f"SPEAKER_{label:02d}",
                'start': metadata[i]['start'],
                'end': metadata[i]['end']
            })
        
        return self._merge(results)

    def _merge(self, segments):
        if not segments: return []
        merged = []
        current = segments[0].copy()
        for nxt in segments[1:]:
            if nxt['speaker'] == current['speaker'] and nxt['start'] <= current['end'] + 0.1:
                current['end'] = nxt['end']
            else:
                merged.append(current)
                current = nxt.copy()
        merged.append(current)
        return merged
