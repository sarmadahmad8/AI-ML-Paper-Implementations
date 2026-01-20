import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model import PSMNet
from data_setup import create_dataloaders_SintelStereo
from utils import save_checkpoint, load_checkpoint

EPOCHS = 20
LR = 5e-4
BATCH_SIZE = 2
HEIGHT, WIDTH = 256, 512
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders_SintelStereo(batch_size= BATCH_SIZE,
                                                                                              num_workers= 8)

psmnet = PSMNet(height=HEIGHT,
                width=WIDTH,
                max_disp=192).to(device)

loss_fn = nn.SmoothL1Loss(reduction="mean").to(device)
            
optimizer = torch.optim.Adam(params= psmnet.parameters(),
                             lr= LR,
                             betas= (0.9, 0.999))

load_checkpoint(model= psmnet,
                optimizer= optimizer,
                checkpoint_name= "PSMNet-Sintel-40epochs-Experiment1.pth")

# scheduler = StepLR(optimizer= optimizer,
#                    step_size= 61 * 131,
#                    gamma= 0.1)

results = train_Kitti(model= psmnet,
                       train_dataloader= train_dataloader,
                       test_dataloader= test_dataloader,
                       loss_fn= loss_fn,
                       optimizer= optimizer,
                       scheduler = None,
                       crop= (HEIGHT, WIDTH),
                       epochs=EPOCHS,
                       device= device,
                       use_scaler= True)
