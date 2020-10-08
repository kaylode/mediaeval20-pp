from utils.getter import *


def train():
    trainset = ImageFolder("datasets/koniq/pp2020_dev")
    valset = ImageFolder("datasets/koniq/pp2020_test")
    print(trainset)
    print(valset)

    device = torch.device("cuda")
    BATCH_SIZE = 4
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=trainset.collate_fn, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=trainset.collate_fn, shuffle=False)

    optimizer = torch.optim.Adam

    model = FullModel(
                    optimizer= optimizer,
                    optim_params = {'lr': 1e-3},
                    criterion= None, 
                    device = device)

    #model.load_state_dict(torch.load('/content/drive/My Drive/weights/Pixel Privacy/unet_ssim/unet_ssim_6_    0.8850.pth'))

    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_epoch=5, path = 'weights/cae'),
                     logger = Logger(log_dir='loggers/runs/cae'),
                     scheduler = StepLR(model.optimizer, step_size=30, gamma=0.1),
                     evaluate_per_epoch = 2)

    print(trainer)

    trainer.fit(num_epochs=50, print_per_iter=10)
    
    
    
  
if __name__ == "__main__":
    train()
    