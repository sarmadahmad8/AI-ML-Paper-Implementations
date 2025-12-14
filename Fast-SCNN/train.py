
import argparse
from collections import Counter
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torchvision.transforms import v2
from download_data import download_data
from data_setup import choose_dataloader
from model import FastSCNN
from engine import train_CS_or_ADE
from utils import load_checkpoint, save_checkpoint, evaluate_model_multiclass

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type= str, help="Provide name of dataset you want to load (e.g 'Cityscape', 'ADE20K'.")
parser.add_argument("--batch_size", type=int, help="Provide batch size to use for dataloading (e.g '4', '8', '16').")
parser.add_argument("--num_workers", type=int, help="Provide number of cpu cores to use for dataloading (e.g '4', '8', '16'). '4' is recommended.")
parser.add_argument("--sample_size", type=float, help="Provide ratio of data to use for training (e.g '0.2', '0.5', '1.0').")
parser.add_argument("--epochs", type=int, help="Provide the number of epochs to train for (e.g '5', '10', '100', '1000').")
parser.add_argument("--load_checkpoint", type=str, help="Provide the checkpoint name to load. The model should be in the ./models folder and the optimizer should be in ./optimizers folder.")
parser.add_argument("--save_checkpoint", type=str, help="Provide a name to save the checkpoint with. The name should end with either '.pt' or '.pth'")

args = parser.parse_args()
data_path = download_data(dataset_name = args.dataset_name)

img_transforms = v2.Compose([
    v2.RandomResizedCrop(size= (1024, 2048),
                         scale=(0.5, 2.0),
              interpolation= v2.InterpolationMode.NEAREST),
    v2.RandomAffine(degrees= 0,
                    translate= (0.05, 0.10),
                    interpolation= v2.InterpolationMode.NEAREST),
    v2.RandomHorizontalFlip(p=0.2),
    v2.ToTensor()
])

if args.dataset_name == "Cityscape":
    remove_classes=[0, 2, 5, 10, 14, 15, 16, 18, 19, 25, 27, 29, 30, 31, 32]
    remap_classes= {-1: -1, 4: 0, 6: 1, 7: 2, 8: 3, 9: 4, 11: 5, 12: 6, 13: 7, 17: 8, 20: 9, 21: 10, 22: 11, 23: 12, 24: 13, 26: 14, 28: 15, 33: 16}
    num_classes = 18

else:
    remove_classes= [34, 35, 41, 44, 46, 49, 51, 53, 54, 57, 60, 61, 62, 63, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]
    remap_classes= {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 36: 34, 37: 35, 38: 36, 39: 37, 40: 38, 42: 39, 43: 40, 45: 41, 47: 42, 48: 43, 50: 44, 52: 45, 55: 46, 56: 47, 58: 48, 59: 49, 64: 50, 65: 51, 66: 52, 71: 53, 78: 54, 79: 55, 82: 56, 106: 57, 120: 58}
    num_classes = 60
    

train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = choose_dataloader(dataset_name = args.dataset_name,
                                                                                                                data_path = data_path,
                                                                                                                img_transforms = img_transforms,
                                                                                                                batch_size = args.batch_size,
                                                                                                                num_workers = args.num_workers,
                                                                                                                sample_size = args.sample_size,
                                                                                                                remove_classes = remove_classes,
                                                                                                                remap_classes = remap_classes)
# Count labels
class_count = Counter()
counts_only_ascending = []
print(f"Calculating class weights....")
for i in tqdm(range(len(train_dataset))):
    X, y = train_dataset[i]
    class_count.update(y.flatten().tolist())

sorted_class_count = sorted(class_count.items())

for class_idx, count in sorted_class_count:
    counts_only_ascending.append(count)

# Calculate weights
pixel_counts = torch.tensor(counts_only_ascending, dtype = torch.float32)

weights = 1 / pixel_counts
weights = weights / weights.sum()
weights = weights * len(weights)
weights = torch.clamp(weights, max=10.0)


EPOCHS = args.epochs
BATCHES = len(train_dataloader)
device = "cuda" if torch.cuda.is_available() else "cpu"

fast_scnn = FastSCNN(in_channels=3,
                     out_channels=num_classes,
                     expansion_factor= 6)

decay_params, no_decay_params = [], []
for name, param in fast_scnn.named_parameters():
    if not param.requires_grad:
        continue
    if "pointwise" in name:
        decay_params.append(param)
    else:
        no_decay_params.append(param)

loss_fn = nn.CrossEntropyLoss(ignore_index=-1,
                              weight= weights.to(device))

optimizer = torch.optim.SGD(params = [{"params":decay_params, "weight_decay":0.00004},
                                      {"params": no_decay_params, "weight_decay": 0.0}],
                            lr = 0.01,
                            momentum= 0.9)

if args.load_checkpoint:
    fast_scnn, optimizer = load_checkpoint(model= fast_scnn,
                                           optimizer= optimizer,
                                           checkpoint_name=args.load_checkpoint,
                                           device= device)

sceduler = torch.optim.lr_scheduler.PolynomialLR(optimizer= optimizer,
                                                 power= 0.9,
                                                 total_iters= EPOCHS * BATCHES)


results = train_CS_or_ADE(model= fast_scnn,
                           train_dataloader= train_dataloader,
                           test_dataloader= val_dataloader,
                           loss_fn= loss_fn,
                           optimizer= optimizer,
                           scheduler= sceduler,
                           device= device,
                           epochs= EPOCHS,
                           num_classes= num_classes)

if args.save_checkpoint:
    save_checkpoint(model = fast_scnn,
                    optimizer = optimizer,
                    checkpoint_name = args.save_checkpoint,
                    model_target_dir = "models",
                    optimizer_target_dir = "optimizers")
else:
    save_checkpoint(model = fast_scnn,
                    optimizer = optimizer,
                    checkpoint_name = f"Fast-SCNN_{args.dataset_name}_{args.sample_size}_{args.epochs}.pth",
                    model_target_dir = "models",
                    optimizer_target_dir = "optimizers")

results = evaluate_model_multiclass(model = fast_scnn,
                                    dataloader = val_dataloader,
                                    loss_fn = loss_fn,
                                    device = device)
