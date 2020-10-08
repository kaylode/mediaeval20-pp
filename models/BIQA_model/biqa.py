from .inceptionresnetv2 import inceptionresnetv2
import torch.nn as nn
import torch 

class model_qa(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(model_qa,self).__init__()
        base_model = inceptionresnetv2(num_classes=1000, pretrained=None)
        self.base= nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),         
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BIQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = model_qa(num_classes = 1)
        self.net.load_state_dict(torch.load('models\BIQA_model\KonCept512.pth'))

        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

    def forward(self, x):
        return self.net(x)