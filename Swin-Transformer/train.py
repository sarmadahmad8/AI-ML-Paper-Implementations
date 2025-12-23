from engine import train
from model import swin_t
from utils import load_checkpoint, save_checkpoint, plot_metrics
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

img_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p= 0.5),
                                     transforms.RandomRotation(degrees=270),
                                     transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor()
                                    ])

train_dataset = CIFAR100(root= "../data/CIFAR100",
                         train= True,
                         transform= img_transforms,
                         download= True)

test_dataset = CIFAR100(root= "../data/CIFAR100",
                        train= False,
                        transform= img_transforms,
                        download= True)


train_dataloader = DataLoader(dataset= train_dataset,
                              batch_size= 64,
                              num_workers= 8,
                              shuffle= True)

test_dataloader = DataLoader(dataset= test_dataset,
                             batch_size= 64,
                             num_workers= 8,
                             shuffle= False)
print(len(train_dataloader))
device = "cuda" if torch.cuda.is_available() else "cpu"


swin_t_model = swin_t(hidden_dim = 96,
                      layers = (2, 2, 2, 2),
                      heads = (3, 6, 12, 24),
                      in_channels = 3,
                      num_classes = 100,
                      head_dim = 32,
                      window_size = 7,
                      downscaling_factor = (4, 2, 2, 2),
                      relative_pos_embedding = True)

swin_t_model = torch.compile(model= swin_t_model)
swin_t_model.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params= swin_t_model.parameters(),
                            lr= 0.01,
                            momentum=0.9)


load_checkpoint(model= swin_t_model,
                optimizer= optimizer,
                checkpoint_name="Swin-Transformer-CIFAR100-25epochs.pth")


results = train(model= swin_t_model,
                train_dataloader= train_dataloader,
                test_dataloader= test_dataloader,
                device= device,
                loss_fn= loss_fn,
                optimizer= optimizer,
                epochs= 5)

save_checkpoint(model= swin_t_model,
                optimizer = optimizer,
                checkpoint_name = "Swin-Transformer-CIFAR100-30epochs.pth")

plot_metrics(results= results)
