import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model import PWCNet
from PWCNet import PWCDCNet_old
from loss import PWCNetLoss
from data_setup import create_dataloaders_Sintel
from utils import load_checkpoint
        
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders_Sintel(batch_size= BATCH_SIZE,
                                                                                           num_workers= 8)

# pwc_net = PWCDCNet_old().to(device)
pwc_net = PWCNet().to(device)

loss_fn = PWCNetLoss().to(device)
            
optimizer = torch.optim.Adam(params= pwc_net.parameters(),
                             lr= LR,
                             betas= (0.9, 0.999))

load_checkpoint(model= pwc_net,
                optimizer= optimizer,
                checkpoint_name= "PWCNet-Sintel-5epochs-Experiment3.pth")

scheduler = StepLR(optimizer= optimizer,
                   step_size= 61 * 131,
                   gamma= 0.1)

results = train_Sintel(model= pwc_net,
                       train_dataloader= train_dataloader,
                       test_dataloader= test_dataloader,
                       loss_fn= loss_fn,
                       optimizer= optimizer,
                       scheduler = scheduler,
                       resize= (448, 1024),
                       epochs=EPOCHS,
                       device= device,
                       use_scaler= True)
