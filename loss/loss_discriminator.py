import torch
import torch.nn as nn

class LossDSCreal(nn.Module):
    """
    Inputs: r
    """
    def __init__(self):
        super(LossDSCreal, self).__init__()
        
    def forward(self, r):
        loss = torch.max(torch.zeros_like(r), 1 - r)
        return loss.mean()

class LossDSCfake(nn.Module):
    """
    Inputs: r, rhat
    """
    def __init__(self):
        super(LossDSCfake, self).__init__()
        
    def forward(self, rhat):
        loss = torch.max(torch.zeros_like(rhat),1 + rhat)
        return loss.mean()