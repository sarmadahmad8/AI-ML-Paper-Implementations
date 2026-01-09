import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model import SPyNet, ConvNet
from loss import EndPointErrorLoss
from data_setup import create_dataloaders_Sintel
from utils import load_checkpoint
        
EPOCHS = 150
LR = 1e-4
BATCH_SIZE = 6
LAYERS = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders_Sintel(batch_size= BATCH_SIZE,
                                                                                           num_workers= 8)

spynet = SPyNet(layers=LAYERS).to(device)
# convnet = ConvNet()
loss_fn = EndPointErrorLoss().to(device)

for i, convnet_module in enumerate(spynet.spy_net):
    if i >= (LAYERS - 1):
        checkpoint_path = f"models/ConvNet-{i}-Sintel-150epochs-3experiment.pth"
        convnet_module.load_state_dict(torch.load(checkpoint_path))
    else:
        checkpoint_path = f"models/ConvNet-{i+1}-Sintel-150epochs-3experiment.pth"
        convnet_module.load_state_dict(torch.load(checkpoint_path))

    if i != (len(spynet.spy_net) - 1):
        for param in convnet_module.parameters():
            param.requires_grad = False
            
optimizer = torch.optim.Adam(params= filter(lambda p: p.requires_grad, spynet.parameters()),
                             lr= LR,
                             betas= (0.9, 0.999))

scheduler = StepLR(optimizer= optimizer,
                   step_size= 61 * 174,
                   gamma= 0.1)

results = train_Sintel(model= spynet,
                       train_dataloader= train_dataloader,
                       test_dataloader= test_dataloader,
                       loss_fn= loss_fn,
                       optimizer= optimizer,
                       scheduler = scheduler,
                       resize= (448, 1024),
                       epochs=EPOCHS,
                       device= device)
