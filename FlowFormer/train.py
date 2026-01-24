import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model import RAFT
from loss import RAFTLoss
from data_setup import create_dataloaders_Sintel
from engine import train_Sintel
from utils import load_checkpoint
        
EPOCHS = 20
LR = 4e-4
BATCH_SIZE = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders_Sintel(batch_size= BATCH_SIZE,
                                                                                           num_workers= 8, 
                                                                                          sample_size=1.0)
raft = RAFT(hidden_dim=128,
            contest_dim=128,
            iters=32).to(device)

loss_fn = RAFTLoss().to(device)
            
optimizer = torch.optim.AdamW(params= raft.parameters(),
                             lr= LR,
                             betas= (0.9, 0.999))

# load_checkpoint(model= raft,
#                 optimizer= optimizer,
#                 checkpoint_name= "RAFT-Sintel-20epochs-Experiment1.pth")

scheduler = StepLR(optimizer= optimizer,
                   step_size= 10 * 936,
                   gamma= 0.5)

results = train_Sintel(model= raft,
                       train_dataloader= train_dataloader,
                       test_dataloader= test_dataloader,
                       loss_fn= loss_fn,
                       optimizer= optimizer,
                       scheduler = scheduler,
                       resize= (368, 768),
                       epochs=EPOCHS,
                       device= device,
                       use_scaler= True)
