from utils.getter import *
import argparse
from models.BIQA_model.biqa import BIQA
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms
from PIL import Image

torch.backends.cudnn.fastest = True
torch.backends.cudnn.benchmark = True

def visualize(x_ori, x_adv, ori_pred, adv_pred, output_path, idx, debug):
    
    adv_score = np.round(adv_pred[0][0], 5)
    ori_score = np.round(ori_pred[0][0], 5)
    x_adv = denormalize(x_adv.cpu().squeeze(0))
    x_ori = denormalize(x_ori.cpu().squeeze(0))
    

    if debug:
        Image.fromarray((x_adv*255).astype(np.uint8)).save(os.path.join(output_path, f'[{adv_score}]_{idx}.png'))
        return

    fig = plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.imshow(x_ori)
    plt.title(ori_score)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(x_adv)
    plt.title(adv_score)
    
    plt.axis('off')
    plt.savefig(os.path.join(output_path, f'[{adv_score}]_{idx}.png'))
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

def attack(args, config):
    device = torch.device("cuda" if args.cuda else "cpu") #
    
    if args.images is not None:
        input_path, output_path = args.images.split(':')
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    valset = ImageFolder(input_path, args.enhance)
    valloader = data.DataLoader(valset, batch_size=1, collate_fn=valset.collate_fn, shuffle=False, pin_memory=True, num_workers=4)
    
    model = BIQA().to(device)
    loss = nn.MSELoss()

    # Hyper parameters
    y_target = Variable(torch.FloatTensor([[config.attack_label]]), requires_grad=False).to(device)    #9= ostrich
    epsilon = config.epsilon #0.05
    num_steps = config.num_steps #20
    alpha = config.alpha #0.05

    scores_list = []
    for idx, batch in enumerate(tqdm(valloader)):
        image_tensor_ori = batch['imgs'].to(device)
        image_tensor = batch['labels'].to(device) 
        img_variable = Variable(image_tensor, requires_grad=True).to(device)
        img_variable.data = image_tensor
        
        if config.brute_force:
            num_steps = config.max_steps

        for i in range(num_steps):
            zero_gradients(img_variable)
            output = model(img_variable)
            
            loss_cal = loss(output, y_target)
            loss_cal.backward()
            x_grad = alpha * torch.sign(img_variable.grad.data)
            adv_temp = img_variable.data - x_grad
            total_grad = adv_temp - image_tensor
            total_grad = torch.clamp(total_grad, -epsilon, epsilon)
            img_variable.data = image_tensor + total_grad
            if config.brute_force:
                if i % int(config.max_steps/5) == 0:
                    epsilon += 0.05
                score_adv = model(img_variable).cpu().detach().item()
                if score_adv <= config.max_score:
                    epsilon = config.epsilon
                    break
                    
        with torch.no_grad():
            score_adv = model(img_variable).cpu().detach().numpy()
            score_ori = model(image_tensor_ori).cpu().detach().numpy()

        scores_list += score_adv.reshape(-1).tolist()
        visualize(image_tensor_ori, img_variable.data, score_ori, score_adv, output_path, str(idx).zfill(5), args.debug)
    plot_score(scores_list, output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training on Koniq-10k')
    parser.add_argument('--images', type=str, 
                        help='path to image folder')
    parser.add_argument('--enhance', type=str, default=None,
                        help='path to enhanced image folder')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Using GPU')
    parser.add_argument('--debug',
                        help='save comparison and distribution', action='store_true')
    parser.add_argument('--config', default='config/fgsm.yaml',
                        help='yaml config')
    args = parser.parse_args() 
    
    config = Config(args.config)
    attack(args, config)