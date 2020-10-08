import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .unet import U_Net
from .BIQA_model.biqa import BIQA

from losses.ssimloss import SSIM

class FullModel(BaseModel):
    def __init__(self, alpha=0.1, hard_label=0, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'full_model'
        self.alpha = alpha
        self.hard_label = hard_label

        self.reconstruct = U_Net(3,3)
        self.evalute = BIQA()

        self.ssim_loss = SSIM()
        self.mae_loss = nn.SmoothL1Loss()

        self.inference_mode = False
        
        self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
        self.set_optimizer_params()
        self.n_classes = 1

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False
  
        if self.device:
            self.to(self.device)

    def __str__(self):
        s1 = f'UNet: {self.count_parameters(self.reconstruct)}\n'
        s2 = f'BIQA: {self.count_parameters(self.evaluate)}\n'
        s3 = f'Total: {self.count_parameters(self)}'
        return s1 + s2 + s3

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def zero_grad(self):
        for params in self.parameters():
            params.grad = None

    def train(self):
        self.reconstruct.train()
    
    def eval(self):
        self.reconstruct.eval()

    def inference(self):
        self.inference_mode = True

    def forward(self, inputs):
        reconstructed = self.reconstruct(inputs)
        evaluated = self.evalute(reconstructed)
        if self.inference_mode:
            return reconstructed, evaluated
        else:
            reconstructed_loss = -self.ssim_loss(reconstructed, inputs) + self.mae_loss(reconstructed, inputs)

            fake_labels = (self.hard_label*torch.ones(evaluated.shape)).to(evaluated.device)
            attack_loss = self.mae_loss(evaluated, fake_labels)
            total_loss = self.alpha*attack_loss + reconstructed_loss
            return  total_loss, {'SSIM': abs(reconstructed_loss.item()), 'MAE': attack_loss.item(), 'T': abs(total_loss.item())}

    
    def training_step(self, batch):
        inputs = batch["imgs"]
        targets = batch["labels"]
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        
        loss = self(inputs)
        return loss

    
    def inference_step(self, batch):
        inputs = batch['imgs']
        if self.device:
            inputs = inputs.to(self.device)
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=1)

        if self.device:
            preds = preds.cpu()
        return preds.numpy()

    def evaluate_step(self, batch):
        inputs = batch["imgs"]
        targets = batch["labels"]
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        loss = self(inputs) #batchsize, label_dim
        

        metric_dict = {}
        return loss , metric_dict