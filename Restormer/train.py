import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Restormer
from engine import train
from download_data import download_data
from data_setup import create_dataloaders_GoPro, create_dataloaders_Urban100, create_dataloaders_DIV2K
from utils import load_checkpoint, save_checkpoint, evaluate_model, plot_reconstructed_images, plot_metrics

ITERATIONS = 92000
DISCRIMINATOR = False
BATCH_SIZE = 2
PATCH_SIZE = 128
CROPS_PER_IMAGE = (ITERATIONS * BATCH_SIZE) // 800
SAMPLE_SIZE = 1.0
LR = 3e-4
EPOCHS = 20

gopro_path = download_data(dataset_name="Div2K")
# urban100_path = download_data(dataset_name="Urban100")

train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloaders_DIV2K(img_dir= gopro_path,
                                                                                                                      sample_size= SAMPLE_SIZE,
                                                                                                                      batch_size= BATCH_SIZE,
                                                                                                                      scale= 1,
                                                                                                                      crops_per_image= CROPS_PER_IMAGE,
                                                                                                                      add_noise= True,
                                                                                                                      sigma= 0.2,
                                                                                                                      patch_size_lr= PATCH_SIZE,
                                                                                                                      test_val_split= 0.1,
                                                                                                                      num_workers=8)

print(len(train_dataloader), len(test_dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"

restormer = Restormer(in_channels=3,
                      out_channels=3,
                      in_dimensions=(48, 96, 192, 384),
                      t_layers=(4, 6, 6, 8),
                      heads=(1, 2, 4, 8),
                      gamma= 2.666,
                      dropout= 0.1,
                      drop_path_prob= None).to(device)

# restormer = torch.compile(model= restormer)

loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(params= restormer.parameters(),
                             lr= LR,
                             betas=(0.9, 0.999),
                             weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer= optimizer,
                                            step_size= 92000,
                                            gamma= 1.0)

# load_checkpoint(model= restormer,
                # optimizer= optimizer,
                # scheduler= scheduler,
#                 checkpoint_name=f"Restormer-GoPro-20000iterations-L1-64-no_skip_con.pth")

results = train(model= restormer,
            train_dataloader= train_dataloader,
            test_dataloader= test_dataloader,
            loss_fn= loss_fn,
            optimizer= optimizer,
            scheduler= scheduler,
            device= device,
            experiment_name = "residual_learning",
            amp= True)


plot_metrics(results= results)

plot_reconstructed_images(model= restormer,
                          val_dataset= val_dataset,
                          samples= 5,
                          save_name=f"Restormer-GoPro-{ITERATIONS}epochs-{CROPS_PER_IMAGE}crops-{PATCH_SIZE}patch-L1")

save_checkpoint(model= restormer,
                optimizer= optimizer,
                scheduler= scheduler,
                checkpoint_name= f"Restormer-GoPro-{ITERATIONS}epochs-{CROPS_PER_IMAGE}crops-{PATCH_SIZE}patch-L1.pth")
