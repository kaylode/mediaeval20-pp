import sys
sys.path.append("..")

import torch
import numpy as np
from losses.ssimloss import ssim

class SSIMMetric():
    """
    Accuracy metric for classification
    """
    def __init__(self, decimals = 10):
        self.reset()
        self.decimals = decimals

    def compute(self, output, target):
        scores = ssim(output, target)
        return scores.cpu().numpy()

    def update(self,  output, target):
        assert isinstance(output, torch.Tensor), "Please input tensors"
        scores = self.compute(output, target)
        self.scores_list.append(scores)
        self.sample_size += 1 #score compute for a batch

    def reset(self):
        self.scores_list = []
        self.sample_size = 0

    def value(self):
        values = sum(np.array(self.scores_list)) / self.sample_size

        return {"ssim" : np.around(values, decimals = self.decimals)}

    def __str__(self):
        return f'SSIM Score: {self.value()}'

    def __len__(self):
        return len(self.sample_size)

if __name__ == '__main__':
    ssim_score = SSIMMetric(decimals = 4)
    imgs = torch.rand(3,3,256,256).cuda()
    targets = imgs.clone()
    imgs[0] = torch.rand(3,256,256).cuda()
    
    ssim_score.update(imgs, targets)
    di = {}
    di.update(ssim_score.value())
    print(di)
    
  
