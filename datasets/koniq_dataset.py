import torch
import torch.utils.data as data
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms as TF

def denormalize(img, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]): 
    mean = np.array(mean)
    std = np.array(std)
    img_show = img.clone()
    if img_show.shape[0] == 1:
        img_show = img_show.squeeze(0)
    img_show = img_show.numpy().transpose((1,2,0))
    img_show = (img_show * std+mean)
    img_show = np.clip(img_show,0,1)
    return img_show


class ImageFolder(data.Dataset):
    def __init__(self, path, enhance_path = None):
        self.path = path
        self.enhance_path = enhance_path if enhance_path is not None else path
        self.transforms = TF.Compose([
            TF.CenterCrop((384, 512)),
            TF.ToTensor(),
            TF.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.load_data()
        
    def load_data(self):
        self.fns = [(os.path.join(self.path, i), os.path.join(self.enhance_path, i)) for i in os.listdir(self.path)]
        
    def __getitem__(self, idx):
        item = self.fns[idx]
        img = Image.open(item[0]).convert('RGB')
        label = Image.open(item[1]).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
            label = self.transforms(label)

        return {'img': img, 'label': label}
    
    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes by index
        """

        if index is None:
            index = np.random.randint(0,len(self.fns))
        item = self.__getitem__(index)
        img = item['img']
        label = item['label']

        # Denormalize and reverse-tensorize
        img = denormalize(img)

        label = label.numpy()
        self.visualize(img, label, figsize = figsize)

    
    def visualize(self, img, label, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes
        """
        fig,ax = plt.subplots(figsize=figsize)

        # Display the image
        ax.imshow(img)
        plt.title(self.classes[label])
        
        plt.show()

    def __len__(self):
        return len(self.fns)
    
    def __str__(self):
        s = "Koniq-10K Dataset \n"
        line = "-------------------------------\n"
        s1 = "Number of samples: " + str(len(self.fns)) + '\n'
        return s + line + s1

    def collate_fn(self, batch):
        imgs = torch.stack([i['img'] for i in batch])
        labels = torch.stack([i['label'] for i in batch])
        return {'imgs': imgs, 'labels': labels}