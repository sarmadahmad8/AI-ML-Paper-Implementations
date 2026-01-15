import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from model import FRVSR, init_params
from data_setup import create_dataloaders_custom, create_dataloaders_vid4
from engine import train
from utils import load_checkpoint, save_checkpoint, plot_reconstructed_images, evaluate_model
from loss import FRVSRLoss


SCALE = 4
BATCH_SIZE = 2
SAMPLE_SIZE = 0.01
LR = 1e-4
PRETRAINED_EPOCHS = 10
USE_AMP = False
EPOCHS = 10

if USE_AMP:
    PRECISION = "fp16"
else:
    PRECISION = "fp32"

img_dir = Path("../data/VSRDataset/")
train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders_custom(img_dir=img_dir, 
                                                                                           sample_size=SAMPLE_SIZE,
                                                                                           batch_size=BATCH_SIZE)

print(len(train_dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"

frvsr = FRVSR(res_in_channels=128,
                residual_blocks=10,
                scale_factor=4).apply(init_params).to(device)

loss_fn = FRVSRLoss(reduction="sum").to(device)

optimizer = torch.optim.Adam(params= frvsr.parameters(),
                             lr= LR,
                             betas=(0.9, 0.999))
load_checkpoint(model= frvsr,
                optimizer= optimizer,
                checkpoint_name= f"FRVSR-CustomDataset-{PRETRAINED_EPOCHS}epochs-{PRECISION}.pth")

results = train(model= frvsr,
                train_dataloader= train_dataloader,
                test_dataloader= test_dataloader,
                loss_fn= loss_fn,
                optimizer= optimizer,
                scheduler=None,
                device= device,
                epochs= EPOCHS,
                pretrained_epochs=PRETRAINED_EPOCHS,
                use_amp= False)

img_dir = Path("../data/Vid4/")
val_dataloader, val_dataset = create_dataloaders_vid4(img_dir=img_dir,
                                                      batch_size=4)

evaluate_model(model= frvsr,
               loss_fn=loss_fn,
               val_dataloader=val_dataloader,
               crop_border=4,
               device=device)

plot_reconstructed_images(model=frvsr,
                          val_dataset=val_dataset,
                          samples = 2,
                          device= "cpu",
                          save_name=f"FRVSR-Vid4-{EPOCHS + PRETRAINED_EPOCHS}epochs-{PRECISION}-{SAMPLE_SIZE}samplesize-Evaluation")
