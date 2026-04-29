import torch
import numpy as np
import torchaudio
import os
from models.transformer_diarization import DiarizationTransformer
from .embedding import SpeakerEmbedding
from .clustering import SpeakerClustering
from .vad import VAD # Import classic VAD as backup

class HybridPipeline:
    def __init__(self, model_path=None, n_speakers=None, device=None):
        # 1. Automatic Device Detection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Optional fixed speaker count
        self.n_speakers = n_speakers
        
        # 2. Dynamic Memory Allocation Strategy
        self.chunk_size_sec = 30 # Default
        if "cuda" in self.device:
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) # In GiB
            # Scale chunk size based on VRAM (roughly 10 sec per GB of free-ish space)
            if total_vram <= 4:
                self.chunk_size_sec = 15 # Aggressive for 4GB
                print(f"Low VRAM detected ({total_vram:.2f}GB). Using micro-chunks (15s).")
            elif total_vram <= 8:
                self.chunk_size_sec = 30
                print(f"Mid VRAM detected ({total_vram:.2f}GB). Using standard chunks (30s).")
            else:
                self.chunk_size_sec = 120 # Fast for 20GB+
                print(f"High VRAM detected ({total_vram:.2f}GB). Using large chunks (120s).")
        
        # Neural Model
        if model_path and os.path.exists(model_path):
            # 1. Detect n_speakers from checkpoint to avoid size mismatch
            detected_n_spk = n_speakers
            try:
                ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                if 'classifier.weight' in ckpt:
                    detected_n_spk = ckpt['classifier.weight'].shape[0]
                    print(f"Detected {detected_n_spk} speakers in checkpoint.")
            except Exception as e:
                print(f"Warning: Could not detect speaker count from checkpoint: {e}")
                detected_n_spk = n_speakers if n_speakers else 4

            self.neural_model = DiarizationTransformer(n_speakers=detected_n_spk).to(self.device)
            self.neural_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.neural_model.eval()
            self.has_neural = True
        else:
            self.has_neural = False
        
        # Classic Components
        self.embedder = SpeakerEmbedding(device=self.device)
        self.classic_vad = VAD(threshold=0.3)
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=80
        ).to(self.device)

    def process(self, audio_path, threshold=0.35, num_speakers=None):
        target_speakers = num_speakers if num_speakers is not None else self.n_speakers

        # 1. Load and Preprocess
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        wav_data = waveform.squeeze()
        if wav_data.abs().max() > 0:
            wav_data = wav_data / wav_data.abs().max()

        all_stream_segments = [] # Will store (slot_id, segments)

        # 2. Dynamic Neural Pass (Independent Slot Processing for Overlap)
        if self.has_neural:
            try:
                all_probs = []
                samples_per_chunk = self.chunk_size_sec * 16000
                
                for i in range(0, len(wav_data), samples_per_chunk):
                    chunk = wav_data[i : i + samples_per_chunk].unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        mels = self.feature_extractor(chunk)
                        features = torch.log1p(mels)
                        preds = self.neural_model(features)
                        all_probs.append(preds[0].cpu())
                    if "cuda" in self.device:
                        torch.cuda.empty_cache()

                probs = torch.cat(all_probs, dim=0).numpy() # (Time, Slots)
                n_slots = probs.shape[1]
                hop_length = 512
                frame_duration = hop_length / 16000
                min_frames = int(0.4 / frame_duration)

                # Process each slot independently to catch overlaps
                for slot in range(n_slots):
                    slot_probs = probs[:, slot]
                    active_frames = slot_probs > threshold
                    
                    slot_segments = []
                    start_f = None
                    for f in range(len(active_frames)):
                        if active_frames[f] and start_f is None:
                            start_f = f
                        elif not active_frames[f] and start_f is not None:
                            if (f - start_f) >= min_frames:
                                slot_segments.append({'start': start_f * frame_duration, 'end': f * frame_duration})
                            start_f = None
                    
                    if slot_segments:
                        all_stream_segments.append(slot_segments)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.chunk_size_sec = max(5, self.chunk_size_sec // 2)
                    return self.process(audio_path, threshold, num_speakers=num_speakers) 
                else: raise e

        # 3. Fallback to Classic VAD if Neural fails
        if not all_stream_segments:
            classic_segs = self.classic_vad.get_speech_segments(wav_data, 16000)
            if classic_segs:
                all_stream_segments = [[{'start': s['start']/16000, 'end': s['end']/16000} for s in classic_segs]]

        if not all_stream_segments: return []

        # 4. Identity Pass: Extract Embeddings per Stream
        all_embeddings = []
        embedding_metadata = [] # (stream_id, segment_index)
        
        window_len = 1.5 
        hop_len = 0.75   
        min_rms = 0.015

        for stream_id, segments in enumerate(all_stream_segments):
            for seg_idx, seg in enumerate(segments):
                seg_duration = seg['end'] - seg['start']
                s_idx_total = int(seg['start'] * 16000)
                e_idx_total = int(seg['end'] * 16000)
                seg_wav = wav_data[s_idx_total:e_idx_total]
                
                if torch.sqrt(torch.mean(seg_wav**2)).item() < min_rms:
                    continue

                # Use sliding window within each segment for higher resolution
                curr_start = seg['start']
                while curr_start + window_len <= seg['end'] + 0.1:
                    s_idx = int(curr_start * 16000)
                    e_idx = int(min(seg['end'], curr_start + window_len) * 16000)
                    if e_idx - s_idx < 4000: break # Skip too small windows
                    
                    win_wav = wav_data[s_idx:e_idx]
                    if torch.sqrt(torch.mean(win_wav**2)).item() > min_rms:
                        emb = self.embedder.extract(win_wav)
                        all_embeddings.append(emb)
                        embedding_metadata.append({
                            'stream': stream_id,
                            'start': curr_start,
                            'end': min(seg['end'], curr_start + window_len)
                        })
                    curr_start += hop_len

        if not all_embeddings: return []

        # 5. Clustering: Map all embeddings to Global Speakers
        clusterer = SpeakerClustering(threshold=0.85)
        embeddings_stack = np.stack(all_embeddings)
        
        if len(all_embeddings) == 1:
            labels = [0]
        else:
            if target_speakers:
                from sklearn.cluster import AgglomerativeClustering
                from sklearn.metrics.pairwise import cosine_distances
                dist_matrix = cosine_distances(embeddings_stack)
                clusterer_fixed = AgglomerativeClustering(
                    n_clusters=target_speakers, metric="precomputed", linkage="average"
                )
                labels = clusterer_fixed.fit_predict(dist_matrix)
            else:
                labels = clusterer.cluster(embeddings_stack)
        
        # 6. Reconstruct Timelines per Speaker
        speaker_timelines = {} # {speaker_id: [segments]}
        for i, label in enumerate(labels):
            spk_id = f"SPEAKER_{label:02d}"
            if spk_id not in speaker_timelines:
                speaker_timelines[spk_id] = []
            speaker_timelines[spk_id].append(embedding_metadata[i])
            
        # Merge segments per speaker
        final_results = []
        for spk_id, segments in speaker_timelines.items():
            segments = sorted(segments, key=lambda x: x['start'])
            if not segments: continue
            
            curr = segments[0]
            for i in range(1, len(segments)):
                nxt = segments[i]
                # Merge if they belong to the same speaker and are adjacent/overlapping
                if nxt['start'] <= curr['end'] + 0.2:
                    curr['end'] = max(curr['end'], nxt['end'])
                else:
                    final_results.append({'speaker': spk_id, **curr})
                    curr = nxt
            final_results.append({'speaker': spk_id, **curr})
            
        return [
            {**r, 'start': round(r['start'], 2), 'end': round(r['end'], 2)} 
            for r in sorted(final_results, key=lambda x: x['start'])
        ]
