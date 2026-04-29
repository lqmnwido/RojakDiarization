import numpy as np
import torch
import torchaudio
import random

def get_random_overlap(duration, overlap_prob=0.3, overlap_ratio=0.5):
    """Determines if and how much two segments should overlap."""
    if random.random() < overlap_prob:
        return duration * overlap_ratio
    return 0

class AudioAugmentor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def add_noise(self, audio, noise_level=0.005):
        noise = torch.randn_like(audio) * noise_level
        return audio + noise

    def apply_reverb(self, audio):
        # Placeholder for RIR (Room Impulse Response) convolution
        # In production, you'd use torchaudio.functional.fftconvolve with an RIR dataset
        return audio

class ConversationMixer:
    def __init__(self, sample_rate=16000, segment_len=20):
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.n_samples = segment_len * sample_rate

    def mix(self, utterances_per_speaker):
        """
        utterances_per_speaker: List of lists [[spk1_u1, spk1_u2], [spk2_u1, ...]]
        Returns: mixed_audio, labels [T, S]
        """
        n_speakers = len(utterances_per_speaker)
        mixed_audio = torch.zeros(self.n_samples)
        # 10ms frame resolution for labels
        n_frames = self.segment_len * 100
        labels = torch.zeros(n_frames, n_speakers)
        
        current_time_sec = 0.5 # start with a small offset
        
        # Simple turn-taking simulation with potential overlaps
        while current_time_sec < self.segment_len - 2:
            speaker_idx = random.randint(0, n_speakers - 1)
            if not utterances_per_speaker[speaker_idx]:
                break
                
            u = utterances_per_speaker[speaker_idx].pop(0)
            u_dur = len(u) / self.sample_rate
            
            # Decide overlap with previous turn
            overlap = get_random_overlap(u_dur)
            start_sec = max(0, current_time_sec - overlap)
            
            # Check if we exceed segment length
            if start_sec + u_dur > self.segment_len:
                u_dur = self.segment_len - start_sec
                u = u[:int(u_dur * self.sample_rate)]
            
            start_sample = int(start_sec * self.sample_rate)
            end_sample = start_sample + len(u)
            
            mixed_audio[start_sample:end_sample] += u
            
            # Update labels
            start_frame = int(start_sec * 100)
            end_frame = min(n_frames, start_frame + int(u_dur * 100))
            labels[start_frame:end_frame, speaker_idx] = 1.0
            
            current_time_sec = start_sec + u_dur + random.uniform(0.1, 0.5)

        return mixed_audio, labels
