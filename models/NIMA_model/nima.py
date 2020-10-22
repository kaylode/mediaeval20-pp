from .mobile_net_v2 import MobileNetV2
import numpy as np
import torch
import torch.nn as nn

def get_mean_score(score):
    buckets = np.arange(1, 11)
    mu = (buckets * score).sum()
    return mu

def get_std_score(scores):
    si = np.arange(1, 11)
    mean = get_mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

class model_nima(nn.Module):
    def __init__(self):
        super().__init__()
        base_model =  MobileNetV2()
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class NIMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = model_nima()
        self.net.load_state_dict(torch.load('models/NIMA_model/nima224.pth'))

        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

    def forward(self, x):
        outputs = self.net(x).data.cpu().numpy()[0]
        mean_score = get_mean_score(outputs)
        std_score = get_std_score(outputs)
        return {
            'mean': float(mean_score),
            'std': float(std_score)
        }