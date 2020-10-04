import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, outputs, targets):
        loss =  self.mse(outputs, targets)
        return loss, {"T": loss.item()}