import torch
import torch.nn as nn
import itertools

class PITLoss(nn.Module):
    def __init__(self, base_loss=nn.BCELoss()):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, preds, targets):
        """
        preds: (batch, time, n_speakers)
        targets: (batch, time, n_speakers)
        """
        batch_size, time_steps, n_speakers = targets.shape
        perms = list(itertools.permutations(range(n_speakers)))
        
        batch_losses = []
        
        for b in range(batch_size):
            p_b = preds[b]   # (T, S)
            t_b = targets[b] # (T, S)
            
            perm_losses = []
            for p in perms:
                # Reorder targets according to permutation p
                t_b_perm = t_b[:, p]
                loss = self.base_loss(p_b, t_b_perm)
                perm_losses.append(loss)
            
            # Take the minimum loss among all permutations for this sample
            batch_losses.append(torch.stack(perm_losses).min())
            
        return torch.stack(batch_losses).mean()

class MultiTaskSegmentationLoss(nn.Module):
    def __init__(self, weights={'sad': 1.0, 'scd': 1.0, 'ovd': 1.0}):
        super().__init__()
        self.bce = nn.BCELoss()
        self.weights = weights

    def forward(self, preds, targets):
        loss_sad = self.bce(preds['sad'], targets['sad'])
        loss_scd = self.bce(preds['scd'], targets['scd'])
        loss_ovd = self.bce(preds['ovd'], targets['ovd'])
        
        total_loss = (self.weights['sad'] * loss_sad + 
                      self.weights['scd'] * loss_scd + 
                      self.weights['ovd'] * loss_ovd)
        
        return total_loss, {
            'sad_loss': loss_sad.item(),
            'scd_loss': loss_scd.item(),
            'ovd_loss': loss_ovd.item()
        }
