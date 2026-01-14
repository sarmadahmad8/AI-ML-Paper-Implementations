import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from model import FRVSR, init_params
from data_setup import create_dataloaders
from engine import train
from utils import load_checkpoint, save_checkpoint
from loss import FRVSRLoss


SCALE = 4
BATCH_SIZE = 2
SAMPLE_SIZE = 0.05
LR = 1e-4
PRETRAINED_EPOCHS = 10
EPOCHS = 10

img_dir = Path("../data/VSRDataset/video_1_frames/")
train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders(img_dir=img_dir, 
                                                                                    sample_size=SAMPLE_SIZE,
                                                                                    batch_size=BATCH_SIZE)

print(len(train_dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"

frvsr = FRVSR(res_in_channels=128,
                residual_blocks=10,
                scale_factor=4).apply(init_params).to(device)

# swin_ir = torch.compile(model= swin_ir)

loss_fn = FRVSRLoss().to(device)

optimizer = torch.optim.Adam(params= frvsr.parameters(),
                             lr= LR,
                             betas=(0.9, 0.999))
load_checkpoint(model= frvsr,
                optimizer= optimizer,
                checkpoint_name= f"FRVSR-CustomDataset-{PRETRAINED_EPOCHS}epochs-fp32.pth")

results = train(model= frvsr,
                train_dataloader= train_dataloader,
                test_dataloader= test_dataloader,
                loss_fn= loss_fn,
                optimizer= optimizer,
                scheduler=None,
                device= device,
                epochs= EPOCHS,
                use_amp= False)

save_checkpoint(model= frvsr,
                optimizer= optimizer,
                checkpoint_name= f"FRVSR-CustomDataset-{EPOCHS + PRETRAINED_EPOCHS}epochs-fp32.pth")
