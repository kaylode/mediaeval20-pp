from utils.getter import *
import argparse
from models.BIQA_model.biqa import BIQA
import matplotlib.pyplot as plt
from torchvision import transforms as TF
from PIL import Image
import pandas as pd 

torch.backends.cudnn.fastest = True
torch.backends.cudnn.benchmark = True


transforms = TF.Compose([
            TF.CenterCrop((384, 512)),
            TF.ToTensor(),
            TF.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
   
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

def make_compression(label_paths, transforms):
    out_path = 'results/compressed'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    compressed = []
    for i in label_paths:
        img = Image.open(i)
        img_name = os.path.join(out_path, os.path.basename(i)[:-4]+'.jpeg')
        img.save(img_name, quality=90)
        img = Image.open(img_name)
        compressed.append(img)

    batch = torch.stack([transforms(i) for i in compressed])
    return batch, compressed


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

    out_path = 'results/compressed'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    img_paths = os.listdir(args.images)

    model = BIQA().to(device)

    out_file = {
        'name': [],
        'score': []
    }

    for img_name in tqdm(img_paths):
        img_path = os.path.join(args.images, img_name)

        img = Image.open(img_path)
        img_name = os.path.join(out_path, img_name[:-4]+'.jpeg')
        img.save(img_name, quality=90)
        img = transforms(Image.open(img_name)).unsqueeze(0).to(device)
        
        output = model(img).detach().cpu().numpy()[0]
        out_file['name'].append(img_name)
        out_file['score'].append(output)  
                          
        for i in os.listdir(out_path):
            if i.endswith('.jpeg'):
                os.remove(os.path.join(out_path, i))

    plot_score(out_file['score'], 'results')
    df = pd.DataFrame(out_file, columns=['name', 'score'])
    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training on Koniq-10k')
    parser.add_argument('--images', type=str, 
                        help='path to image folder')
    parser.add_argument('--out_csv', type=str, default=None,
                        help='path to csv output')                    
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Using GPU')
    



    args = parser.parse_args()
    eval(args)
    