import torch
import torch.nn as nn
from model import ResidualChannelAttentionNetwork
from download_data import download_data
from data_setup import create_dataloaders_DIV2K, create_dataloaders_Urban100
from engine import train
from utils import load_checkpoint, save_checkpoint, evaluate_model, plot_reconstructed_images, plot_metrics

SCALE = 2
BATCH_SIZE = 16
CROPS_PER_IMAGE = 15
SAMPLE_SIZE = 1.0
LR = 1e-4
EPOCHS = 20

data_path = download_data(dataset_name="Div2K")

train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloaders_DIV2K(img_dir= data_path,
                                                                                                                          scale= SCALE,
                                                                                                                          sample_size= SAMPLE_SIZE,
                                                                                                                          batch_size= BATCH_SIZE,
                                                                                                                          crops_per_image= CROPS_PER_IMAGE,
                                                                                                                          test_val_split= 0.5)


device = "cuda" if torch.cuda.is_available() else "cpu"

rcan = ResidualChannelAttentionNetwork(num_rg=10,
                                       num_rcab=20,
                                       scale=SCALE).to(device)

# rcan = torch.compile(model= rcan)

loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(params= rcan.parameters(),
                             lr= LR,
                             betas=(0.9, 0.999),
                             eps= 1e-8)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer= optimizer,
                                            step_size=200000,
                                            gamma= 0.5)

# load_checkpoint(model= rcan,
#                 optimizer= optimizer,
#                 scheduler= scheduler,
#                 checkpoint_name="RCAN-DIV2K-20epochs.pth")

results = train(model= rcan,
                train_dataloader= train_dataloader,
                test_dataloader= test_dataloader,
                loss_fn= loss_fn,
                optimizer= optimizer,
                scheduler= scheduler,
                device= device,
                epochs= EPOCHS)

save_checkpoint(model= rcan,
                optimizer= optimizer,
                scheduler= scheduler,
                checkpoint_name= f"RCAN-DIV2K-{EPOCHS}epochs-{CROPS_PER_IMAGE}crops-{SCALE}scale.pth")

plot_metrics(results= results)

evaluate_model(model= rcan,
               loss_fn= loss_fn,
               val_dataloader= val_dataloader,
               device= device)

plot_reconstructed_images(model= rcan,
                          val_dataset= val_dataset,
                          samples= 5,
                          save_name=f"RCAN-DIV2K-{EPOCHS}epochs-{CROPS_PER_IMAGE}crops-{SCALE}scale")
