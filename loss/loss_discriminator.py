import torch
import torch.nn as nn

class LossDSCreal(nn.Module):
    """
    Inputs: r
    """
    def __init__(self):
        super(LossDSCreal, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, r):
        # loss = torch.max(torch.zeros_like(r), 1 - r)
        loss = self.relu(1.0-r)
        return loss.mean()

class LossDSCfake(nn.Module):
    """
    Inputs: rhat
    """
    def __init__(self):
        super(LossDSCfake, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, rhat):
        # loss = torch.max(torch.zeros_like(rhat),1 + rhat)
        loss = self.relu(1.0+rhat)
        return loss.mean()
