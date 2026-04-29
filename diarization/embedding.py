import torch
from speechbrain.inference.speaker import EncoderClassifier

class SpeakerEmbedding:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

    def extract(self, wav):
        """
        Extracts embedding for a single window.
        wav: numpy array or torch tensor
        """
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav).float()
        
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
            
        wav = wav.to(self.device)
        
        with torch.no_grad():
            emb = self.model.encode_batch(wav)
            
        return emb.squeeze().cpu().numpy()
