from utils.getter import *
import argparse
from models.BIQA_model.biqa import model_qa
import matplotlib.pyplot as plt
torch.backends.cudnn.fastest = True
torch.backends.cudnn.benchmark = True

def visualize_test(model, bi, gts, imgs, scores, batch_idx, output_path):

    gts = torch.stack([i for i in gts]).to(imgs.device)

    gt_scores = bi(gts).cpu().numpy()
    pred_scores = bi(imgs).detach().cpu().numpy()

    
    for idx, (gt, pred, gt_score, pred_score) in enumerate(zip(gts,imgs,gt_scores,pred_scores)):
        img_show = denormalize(gt.detach().cpu())
        img_show2 = denormalize(pred.detach().cpu())
        score = np.round(pred_score[0], 4)
        fig = plt.figure(figsize=(8,8))
        plt.subplot(1,2,1)
        plt.title(gt_score[0])
        plt.axis('off')
        plt.imshow(img_show)
        plt.subplot(1,2,2)
        plt.imshow(img_show2)
        plt.title(score)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, f'[{score}]_batch{batch_idx}_{idx}.png'))
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


def eval(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    
    if args.images is not None:
        input_path, output_path = args.images.split(':')
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)

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
        
        bi = model_qa(num_classes=1).to(device)
        bi.load_state_dict(torch.load('/content/drive/My Drive/datasets/Pixel Privacy 2020/KonCept512.pth'))
        bi.eval()
        for param in bi.parameters():
            param.requires_grad = False
        

        scores_list = []


        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valloader)):
                inputs = batch['imgs'].to(device)
                outputs, scores = model(inputs)
                scores_list += scores.cpu().numpy().reshape(-1).tolist()
                visualize_test(model, bi, inputs, outputs, scores, idx, output_path)
                
        plot_score(scores_list, output_path)
    
  
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training on Koniq-10k')
    parser.add_argument('--images', type=str, 
                        help='path to image folder')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Using GPU')
    parser.add_argument('--pretrained', type=str, default= None,
                        help='checkpoint to resume')

    args = parser.parse_args()        
    eval(args)
    