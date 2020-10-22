import torch
import torch.nn as nn
from torchvision import transforms as TF
from models.NIMA_model.nima import NIMA
import argparse
import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 

transforms = TF.Compose([
    TF.Resize((224,224)),
    TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def visualize_test(img, new_img_path, mean_score, std_score):
    mean_score = np.round(mean_score, 3)
    std_score = np.round(std_score, 3)
    title = f'{mean_score} ~ {std_score}'
    plt.title(title)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(new_img_path)
    
   
def plot_score(scores_list, output_path, figsize = (15,15)):
    cnt_dict = {}
    for score in scores_list:
        if int(score) not in cnt_dict.keys():
            cnt_dict[int(score)] = 1
        else:
            cnt_dict[int(score)] += 1
    
    cnt_dict = {k: v for k, v in sorted(cnt_dict.items(), key=lambda item: item[0])}
    fig = plt.figure(figsize = figsize)
    plt.plot(list(cnt_dict.keys()), list(cnt_dict.values()))
    plt.xlabel('Scores')
    plt.ylabel('Number of images')
    plt.title('Score distribution')
    plt.savefig(os.path.join(output_path, f'distribution.png'))
    plt.close(fig)


def eval(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.images is not None:
        input_path, output_path = args.images.split(':')
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    model = NIMA().to(device)

    data = {
        'image name':[], 
        'mean scores':[],
        'std scores': []} 
  
    img_paths = os.listdir(input_path)
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            old_img_path = os.path.join(input_path, img_path)
            new_img_path = os.path.join(output_path, img_path)

            img = Image.open(old_img_path)
            img_tensor = transforms(img).unsqueeze(0).to(device)
            score = model(img_tensor)
            mean_score = score['mean']
            std_score = score['std']

            data['image name'].append(img_path)
            data['mean scores'].append(mean_score)
            data['std scores'].append(std_score)

            visualize_test(img, new_img_path, mean_score, std_score)
        
    if args.csv_output is not None:
        df = pd.DataFrame(data)
        df.to_csv(args.csv_output, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training on Koniq-10k')
    parser.add_argument('--images', type=str, 
                        help='path to image folder')
    parser.add_argument('--csv_output', type=str, default= None,
                        help='path to csv output')            
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Using GPU')

    args = parser.parse_args()        
    eval(args)