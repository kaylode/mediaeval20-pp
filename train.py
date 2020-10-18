from utils.getter import *
import argparse

torch.backends.cudnn.fastest = True
torch.backends.cudnn.benchmark = True

def train(args):
    assert args.reconstruction_loss in ['ssim', 'msssim'], 'wrong reconstruction loss'
    trainset = ImageFolder(os.path.join(args.path, 'pp2020_dev'), os.path.join(args.path,'enhance', 'pp2020_dev'))
    valset = ImageFolder(os.path.join(args.path, 'pp2020_test'), os.path.join(args.path,'enhance', 'pp2020_test'))
    print(trainset)
    print(valset)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, collate_fn=trainset.collate_fn, shuffle=True, pin_memory=True, num_workers=4)
    valloader = data.DataLoader(valset, batch_size=args.batch_size, collate_fn=trainset.collate_fn, shuffle=False, pin_memory=True, num_workers=4)

    model = FullModel(
                    hard_label=args.hard_label,
                    alpha=args.alpha,
                    reconstruction_loss = args.reconstruction_loss,
                    optimizer= torch.optim.Adam,
                    optim_params = {'lr': 1e-3},
                    criterion= None, 
                    device = device)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_epoch=0, save_best = args.save_best, path = args.saved_path),
                     logger = Logger(log_dir=args.log_path),
                     scheduler = StepLR(model.optimizer, step_size=45, gamma=0.1),
                     evaluate_per_epoch = args.val_epoch)

    print(trainer)

    trainer.fit(num_epochs=args.num_epochs, print_per_iter=10)
    
    
    
  
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training on Koniq-10k')
    parser.add_argument('--path', type=str, 
                        help='path to image folder')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Using GPU')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--log_path', type=str, default= None,
                        help='path to tensorboard logging')
    parser.add_argument('--saved_path', type=str, default= None,
                        help='path to saved checkpoint')
    parser.add_argument('--val_epoch', type=int, default= 1,
                        help='validate per epoch')
    parser.add_argument('--save_best', type=bool, default= True,
                        help='save only best ')
    parser.add_argument('--num_epochs', type=int, default= 100,
                        help='number of epochs to train')
    parser.add_argument('--resume', type=str, default= None,
                        help='checkpoint to resume')
    parser.add_argument('--alpha', type=float, default= 0.1,
                        help='alpha for weighting regression loss')
    parser.add_argument('--hard_label', type=float, default= 0,
                        help='hard code label to attack')
    parser.add_argument('--reconstruction_loss', type=str, default= 'ssim',
                        help='reconstruction loss, ssim or msssim')
    args = parser.parse_args()        
    train(args)
    