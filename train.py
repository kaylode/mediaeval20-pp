from utils.getter import *
import torch.nn.functional as F
from losses import MSELoss


data_transforms = Compose([
        #transforms.CenterCrop(),
        Resize((384, 512)),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
              
        return x

class ImageFolder(data.Dataset):
    def __init__(self, path, transforms = None):
        self.path = path
        self.transforms = transforms
        self.load_data()
    
    def load_data(self):
        self.fns = [os.path.join(self.path, i) for i in os.listdir(self.path)]
        
    def __getitem__(self, idx):
        item = self.fns[idx]
        img = Image.open(item).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)['img']

        label = img.detach().clone()
        return img, label
    
    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        labels = torch.stack([i[1] for i in batch])
        return {'imgs': imgs, 'labels': labels}

if __name__ == "__main__":
    
    trainset = ImageFolder("datasets/koni/pp2020_dev", transforms=data_transforms)
    valset = ImageFolder("datasets/koni/pp2020_test", transforms=data_transforms)
    print(len(trainset))
    print(len(valset))
    
    NUM_CLASSES = 1
    device = torch.device("cuda")
    print("Using", device)

    

    # Dataloader
    BATCH_SIZE = 4
    my_collate = trainset.collate_fn
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=False)
    
    criterion = MSELoss()
    optimizer = torch.optim.Adam
    #metrics = [AccuracyMetric(decimals=3)]

    backbone = ConvAutoencoder()
    model = Classifier(
                    n_classes = NUM_CLASSES,
                    backbone = backbone,
                    optim_params = {'lr': 1e-3},
                    criterion= criterion, 
                    optimizer= optimizer,
                    #freeze=True,
                    
                    #metrics=  metrics,
                    device = device)
    
    #load_checkpoint(model, "weights/ssd-voc/SSD300-10.pth")
    #model.unfreeze()
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
#                     clip_grad = 1.0,
                     checkpoint = Checkpoint(save_per_epoch=5, path = 'weights/cae'),
                     logger = Logger(log_dir='loggers/runs/cae'),
                     scheduler = StepLR(model.optimizer, step_size=30, gamma=0.1),
                     evaluate_per_epoch = 2)
    
    print(trainer)
    
    
    
    trainer.fit(num_epochs=50, print_per_iter=10)
    

    # Inference
    """imgs, results = trainer.inference_batch(valloader)
    idx = 0
    img = imgs[idx]
    boxes = results[idx]['rois']
    labels = results[idx]['class_ids']
    trainset.visualize(img,boxes,labels)"""