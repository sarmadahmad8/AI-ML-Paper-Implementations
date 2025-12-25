import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import SwinIR
from download_data import download_data
from data_setup import create_dataloaders_DIV2K, create_dataloaders_Urban100
from engine import train
from utils import load_checkpoint, save_checkpoint, evaluate_model, plot_reconstructed_images, plot_metrics
from loss import L1withPerceptualLoss, L1withPerceptualandGANLoss, Discriminator, CharbonnierLoss

# ITERATIONS = 10000
DISCRIMINATOR = False
SCALE = 1
BATCH_SIZE = 2
CROPS_PER_IMAGE = 4 # (ITERATIONS * BATCH_SIZE) // 800
SAMPLE_SIZE = 1.0
LR = 1e-4
EPOCHS = 10

div2k_path = download_data(dataset_name="Div2K")
# urban100_path = download_data(dataset_name="Urban100")

train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloaders_DIV2K(img_dir= div2k_path,
                                                                                                                      scale= SCALE,
                                                                                                                      sample_size= SAMPLE_SIZE,
                                                                                                                      batch_size= BATCH_SIZE,
                                                                                                                      crops_per_image= CROPS_PER_IMAGE,
                                                                                                                      patch_size_lr= 64,
                                                                                                                      test_val_split= 0.5,
                                                                                                                      add_noise= True,
                                                                                                                      sigma=0.1)

print(len(train_dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"

swin_ir = SwinIR(in_channels=3,
                 out_channels=3,
                 rstb_layers=6,
                 mlp_expansion= 2,
                 swin_t_layers=6,
                 num_heads= 6,
                 head_dim= 30,
                 hidden_dimension= 180,
                 window_size= 8,
                 scale= SCALE,
                 relative_pos_embedding= True).to(device)

# swin_ir = torch.compile(model= swin_ir)

loss_fn = CharbonnierLoss(eps= 1e-3).to(device)

optimizer = torch.optim.Adam(params= swin_ir.parameters(),
                             lr= LR,
                             betas=(0.9, 0.999),
                             eps= 1e-8)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer= optimizer,
                                            step_size=200000,
                                            gamma= 0.5)
if DISCRIMINATOR:
    discriminator = Discriminator(in_channels= 3).to(device)
    disc_optimizer = torch.optim.Adam(params= discriminator.parameters(),
                                 lr= LR,
                                 betas=(0.9, 0.999),
                                 eps= 1e-8)
    
    
    
    disc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer= optimizer,
                                                    step_size=200000,
                                                    gamma= 0.5)

    loss_fn = L1withPerceptualandGANLoss(perceptual_weight = 0.1,
                                         gan_weight = 0.1,
                                         l1_weight = 0.8).to(device)

load_checkpoint(model= swin_ir,
                optimizer= optimizer,
                scheduler= scheduler,
                checkpoint_name=f"SwinIR-DIV2K-10epochs-{CROPS_PER_IMAGE}crops-denoising-charbonnier.pth")

if not DISCRIMINATOR:
    results = train(model= swin_ir,
                    train_dataloader= train_dataloader,
                    test_dataloader= test_dataloader,
                    loss_fn= loss_fn,
                    optimizer= optimizer,
                    scheduler= scheduler,
                    device= device,
                    epochs= EPOCHS)
else:
    results = train(model= swin_ir,
                train_dataloader= train_dataloader,
                test_dataloader= test_dataloader,
                loss_fn= loss_fn,
                optimizer= optimizer,
                scheduler= scheduler,
                device= device,
                epochs= EPOCHS
                discriminator = discriminator,
                disc_optimizer= disc_optimizer,
                disc_scheduler= disc_scheduler)


plot_metrics(results= results)

plot_reconstructed_images(model= swin_ir,
                          val_dataset= val_dataset,
                          samples= 5,
                          save_name=f"SwinIR-DIV2K-{EPOCHS}epochs-{CROPS_PER_IMAGE}crops-{SCALE}scale-L1withPerceptualandGAN")

save_checkpoint(model= swin_ir,
                optimizer= optimizer,
                scheduler= scheduler,
                checkpoint_name= f"SwinIR-DIV2K-{EPOCHS}epochs-{CROPS_PER_IMAGE}crops-{SCALE}scale-L1withPerceptualandGAN.pth")
