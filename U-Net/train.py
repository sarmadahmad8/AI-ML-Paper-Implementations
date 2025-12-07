"""
Script to train and evaluate the UNet model on an input dataset  given some input arguments

Args:
dataset_name: A string containing the name of dataset to download (e.g 'ISBI', 'Cityscape' and 'Carvana').
batch_size: An integer containing the number of images per batch (i.e number of images that pass through the model per train step).
sample_size: A float input containing the ratio of data to use for training (e.g '1' = 100% of the dataset, '0.5' = 50% of the dataset).
num_workers: An integer containing the number of cpu threads to use for dataloading (e.g '4', '12').
use_weight_map: A boolean flag that activates/deactivates the weight map loss computation described in the official conference paper.
load_checkpoint: A string containing the name of the checkpoint (remeber that a checkpoint is both a model and an optimizer disctionary).
checkpoint_name: A string containing the model and optimier name by which you want to save the checkpoint.
epochs: An integer containing the number of epochs you want the model to train for.


"""

from model import UNet
from download_data import download_data
from data_setup import choose_dataloader
import engine
import utils
import evaluate_model
import argparse
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type= str, help= "Provide name of dataset to download as string (e.g 'ISBI', 'Cityscape', 'Carvana')")
# parser.add_argument("--scale_factor", type= float, help= "Provide the scale factor that you would like to scale the data to (e.g '0.5' or '0.25' or '1.0'")
parser.add_argument("--batch_size", type= int, help= "Provide the batch size to train the model on (e.g '1', '4', '8' ...). Provide '1' if 'weight_map' is active.")
parser.add_argument("--sample_size", type= float, help= "Provide the sample size to use from the original large dataset (e.g '0.1', '0.3', '0.5'). Its the percentage of data to use for experiments.")
parser.add_argument("--num_workers", type= int, help="Provide the number of workers to use (e.g '4', '8', '16')")
parser.add_argument("--use_weight_map", action="store_true", help="if flag is used, the loss will be computed using the weight map strategy of the original paper, else simple loss will be computed.")
parser.add_argument("--load_checkpoint", type= str, help= "Provide a valid checkpoint name (model & optimizer) that ends with '.pt' or '.pth'")
parser.add_argument("--checkpoint_name", type= str, help= "Provide a save name for checkpoint (e.g 'UNet-Carvana-20epochs.pth')")
parser.add_argument("--epochs", type= int, help= "Provide the number of epochs to train the model for (e.g '5', '10', '15')")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = download_data(dataset_name=args.dataset_name)

if data_path.stem == "ISBI":
    transforms = transforms.Compose([transforms.Resize(size=(512, 512)),
                                     transforms.ToTensor()
                                    ])
else:
    transforms = transforms.Compose([transforms.Resize(size=(512, 1024)),
                                     transforms.ToTensor()
                                    ])

train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = choose_dataloader(data_path= data_path,
                                                                                                                dataset_name=args.dataset_name,
                                                                                                               transforms=transforms,
                                                                                                               batch_size= args.batch_size,
                                                                                                               num_workers = args.num_workers,
                                                                                                               sample_size = args.sample_size)

print(args.use_weight_map)
if data_path.stem == "ISBI":
    model_0 = UNet(in_channels=1,
                   out_channels=1)
    if args.use_weight_map:
        loss_fn = nn.BCEWithLogitsLoss(reduction = "none")
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    
elif data_path.stem == "CityScape":
    model_0 = UNet(in_channels=3,
                   out_channels=34)
    loss_fn = nn.CrossEntropyLoss()
    
else:
    model_0 = UNet(in_channels=3,
                   out_channels=1)
    if args.use_weight_map:
        loss_fn = nn.BCEWithLogitsLoss(reduction = "none")
    else:
        loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params= model_0.parameters(),
                            lr = 0.01,
                            momentum=0.9)

if args.load_checkpoint:
    model_0, optimizer = utils.load_checkpoint(model= model_0,
                                               optimizer= optimizer,
                                               checkpoint_name= args.load_checkpoint,
                                               device= device)

if data_path.stem == "ISBI":
    results = engine.train_ISBI(model= model_0,
                                train_dataloader= train_dataloader,
                                test_dataloader= test_dataloader,
                                optimizer= optimizer,
                                loss_fn= loss_fn,
                                device= device,
                               epochs=args.epochs,
                               weight_map=args.use_weight_map)
elif data_path.stem == "CityScape":
    results = engine.train_CS(model= model_0,
                                train_dataloader= train_dataloader,
                                test_dataloader= test_dataloader,
                                optimizer= optimizer,
                                loss_fn= loss_fn,
                                device= device,
                             epochs= args.epochs)
else:
    results = engine.train_CV(model= model_0,
                                train_dataloader= train_dataloader,
                                test_dataloader= test_dataloader,
                                optimizer= optimizer,
                                loss_fn= loss_fn,
                                device= device,
                             epochs = args.epochs,
                             weight_map= args.use_weight_map)

if args.checkpoint_name:
    utils.save_checkpoint(model= model_0,
                          optimizer= optimizer,
                          model_target_dir= "models",
                          optimizer_target_dir= "optimizers",
                          checkpoint_name=args.checkpoint_name)

else:
    utils.save_checkpoint(model= model_0,
                          optimizer= optimizer,
                          model_target_dir= "models",
                          optimizer_target_dir= "optimizers",
                          checkpoint_name=f"UNet-{data_path.stem}-{args.sample_size}-{args.epochs}.pth")

if data_path.stem == "ISBI" or data_path.stem == "Carvana":
    evaluate_model.evaluate_model_binaryclass(model= model_0,
                                              dataloader= val_dataloader,
                                              loss_fn= loss_fn,
                                              device= device,
                                             weight_map=args.use_weight_map)

else:
    evaluate_model.evaluate_model_multiclass(model= model_0,
                                          dataloader= val_dataloader,
                                          loss_fn= loss_fn,
                                          device= device)
