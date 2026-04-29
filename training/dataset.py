import torch
from torch.utils.data import Dataset
import torchaudio
import random
from .utils import ConversationMixer, AudioAugmentor

class ProductionDiarizationDataset(Dataset):
    def __init__(self, speaker_data, n_speakers=2, segment_len=20, sample_rate=16000):
        """
        speaker_data: Dict {spk_id: [list of audio paths]}
        """
        self.speaker_data = speaker_data
        self.speaker_ids = list(speaker_data.keys())
        self.n_speakers = n_speakers
        self.mixer = ConversationMixer(sample_rate, segment_len)
        self.augmentor = AudioAugmentor(sample_rate)
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=80
        )

    def __len__(self):
        return 10000 # Virtual size for simulation

    def __getitem__(self, idx):
        # 1. Sample speakers
        selected_spks = random.sample(self.speaker_ids, self.n_speakers)
        
        # 2. Load multiple utterances per speaker
        utterances_per_spk = []
        for spk in selected_spks:
            spk_utts = []
            # Sample 3-5 utterances per speaker to fill the conversation
            paths = random.sample(self.speaker_data[spk], min(5, len(self.speaker_data[spk])))
            for p in paths:
                wav, sr = torchaudio.load(p)
                if sr != self.mixer.sample_rate:
                    wav = torchaudio.transforms.Resample(sr, self.mixer.sample_rate)(wav)
                spk_utts.append(wav.squeeze())
            utterances_per_spk.append(spk_utts)

        # 3. Mix
        audio, labels = self.mixer.mix(utterances_per_spk)
        
        # 4. Augment
        if random.random() < 0.5:
            audio = self.augmentor.add_noise(audio)

        # 5. Extract Features
        features = self.feature_extractor(audio) # [80, T_feat]
        
        return features, labels

def diarization_collate_fn(batch):
    """Handles batching of features and labels."""
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.stack(labels)
    return features, labels
