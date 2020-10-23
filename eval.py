from utils.getter import *
import argparse
from models.BIQA_model.biqa import BIQA
import matplotlib.pyplot as plt
from torchvision import transforms as TF
from PIL import Image
import pandas as pd 

torch.backends.cudnn.fastest = True
torch.backends.cudnn.benchmark = True

def visualize_test(img_paths, enhanced_paths, compressed_paths, output_path, gt_scores, en_scores, com_scores,  debug):
    for (i, j, compressed_show, gt_score, en_score, com_score)  in zip(img_paths, enhanced_paths, compressed_paths, gt_scores, en_scores, com_scores):
        if isinstance(i, str):
            img_name = os.path.basename(i)
            img_show = Image.open(i)
        else:
            img_show = i
        if isinstance(j, str):
            enhanced_show = Image.open(j)
        else:
            enhanced_show = j

        fig = plt.figure(figsize=(15,15))
        
        plt.subplot(1,3,1)
        plt.title('Original: '+ str(gt_score))
        plt.axis('off')
        plt.imshow(img_show)


        plt.subplot(1,3,2)
        plt.imshow(enhanced_show)
        plt.title('Attacked: ' + str(en_score))

        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(compressed_show)
        plt.title('Compressed 90: '+ str(com_score))

        plt.axis('off')
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_path, img_name))
        plt.close(fig)
   
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

def make_compression(label_paths, transforms , label_imgs = None):
    out_path = '/content/drive/My Drive/results/compressed'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    compressed = []
    for idx, i in enumerate(label_paths):
        if label_imgs is None:
          img = Image.open(i)
        else:
          img = Image.fromarray((label_imgs[idx] * 255).astype(np.uint8))
        img_name = os.path.join(out_path, os.path.basename(i)[:-4]+'.jpeg')
        img.save(img_name, quality=90)
        img = Image.open(img_name)
        compressed.append(img)

    batch = torch.stack([transforms(i) for i in compressed])
    return batch, compressed




def eval(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.images is not None:
        input_path, output_path = args.images.split(':')
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)       



        if args.pretrained is not None and args.generated is None:
            valset = ImageFolder(input_path)
            if len(os.listdir(input_path)) > 10:
                batch_size = 4
            else:
                batch_size = 1 
            valloader = data.DataLoader(valset, batch_size=batch_size, collate_fn=valset.collate_fn, shuffle=False, pin_memory=True, num_workers=4)
            print(valset)
            
            model = FullModel(
                            optimizer= torch.optim.Adam,
                            optim_params = {'lr': 1e-3},
                            criterion= None, 
                            device = device)

            if args.pretrained is not None:
                model.load_state_dict(torch.load(args.pretrained))

            model.eval()
            model.inference()
            
            bi = BIQA().to(device)

            for param in bi.parameters():
                param.requires_grad = False
            
            scores_list = []

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(valloader)):
                    inputs = batch['imgs'].to(device)   
                    outputs, en_scores = model(inputs)
                    
                    en_scores = en_scores.detach().cpu().numpy()

                    enhanced = [denormalize(i) for i in outputs.detach().cpu()]
                    compressed, compressed_shows = make_compression(batch['label_paths'], valset.transforms, label_imgs=enhanced)
                    compressed = compressed.to(device)

                    gt_scores = bi(inputs).detach().cpu().numpy()
                    com_scores = bi(compressed).detach().cpu().numpy()

                    scores_list += en_scores.reshape(-1).tolist()

                    visualize_test(batch['img_paths'], enhanced, compressed_shows, output_path, gt_scores, en_scores, com_scores, args.debug)

            for i in os.listdir('/content/drive/My Drive/results/compressed'):
                if i.endswith('.jpeg'):
                    os.remove(os.path.join('/content/drive/My Drive/results/compressed', i))

            plot_score(scores_list, output_path)

        else:
            valset = ImageFolder(input_path, args.generated)
            if len(os.listdir(input_path)) > 10:
                batch_size = 4
            else:
                batch_size = 1 
            valloader = data.DataLoader(valset, batch_size=batch_size, collate_fn=valset.collate_fn, shuffle=False, pin_memory=True, num_workers=4)
    
            model = BIQA().to(device)

            for param in model.parameters():
                param.requires_grad = False

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(valloader)):
                    inputs = batch['imgs'].to(device)
                    enhanced = batch['labels'].to(device)
                    compressed, compressed_shows = make_compression(batch['label_paths'], valset.transforms)
                    compressed = compressed.to(device)
                    gt_scores = model(inputs).detach().cpu().numpy()
                    en_scores = model(enhanced).detach().cpu().numpy()
                    com_scores = model(compressed).detach().cpu().numpy()
                    visualize_test(batch['img_paths'], batch['label_paths'], compressed_shows, output_path, gt_scores, en_scores, com_scores,  debug)
                    
            for i in os.listdir('/content/drive/My Drive/results/compressed'):
                if i.endswith('.jpeg'):
                    os.remove(os.path.join('/content/drive/My Drive/results/compressed', i))
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training on Koniq-10k')
    parser.add_argument('--images', type=str, 
                        help='path to image folder')
    parser.add_argument('--out_csv', type=str, default=None,
                        help='path to csv output')                    
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Using GPU')
    
    parser.add_argument('--debug', action='store_true',
                        help='checkpoint to resume')
    parser.add_argument('--pretrained', type=str, default= None,
                        help='checkpoint to resume')

    parser.add_argument('--generated', type=str, default= None,
                        help='path to generated image')


    args = parser.parse_args()
    eval(args)
    