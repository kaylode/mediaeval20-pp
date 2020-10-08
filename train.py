from utils.getter import *


def train():
    trainset = ImageFolder("datasets/koni/pp2020_dev")
    valset = ImageFolder("datasets/koni/pp2020_test")
    print(len(trainset))
    print(len(valset))
    
    
    device = torch.device("cuda")
    print("Using", device)

    

    # Dataloader
    BATCH_SIZE = 1
    my_collate = trainset.collate_fn
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=False)
    
    
    
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

if __name__ == "__main__":
    trainset = ImageFolder("datasets/koni/pp2020_dev")
    valset = ImageFolder("datasets/koni/pp2020_test")
    print(len(trainset))
    print(len(valset))

    device = torch.device("cuda")
    BATCH_SIZE = 1
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=trainset.collate_fn, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=trainset.collate_fn, shuffle=False)

    optimizer = torch.optim.Adam

    model = FullModel(
                    optimizer= optimizer,
                    optim_params = {'lr': 1e-3},
                    criterion= None, 
                    device = device)

    