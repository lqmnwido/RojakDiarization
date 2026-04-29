import torch
# Note: silero-vad is usually installed via pip, but often used via torch.hub for convenience
# We will use the simplified interface provided by the library if available, or fallback to hub.

class VAD:
    def __init__(self, threshold=0.5):
        # Loading from torch hub as it's the most common "open-source" way to get Silero
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (self.get_speech_timestamps, _, _, _, _) = utils
        self.threshold = threshold

    def get_speech_segments(self, wav, sr):
        """
        Returns a list of dicts with 'start' and 'end' in samples.
        """
        if isinstance(wav, list):
            wav = torch.tensor(wav)
        if wav.ndim > 1:
            wav = wav.squeeze()
            
        speech_timestamps = self.get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=sr,
            threshold=self.threshold
        )
        return speech_timestamps
