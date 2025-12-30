import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2

def binarize_image(image_tensor: torch.Tensor) -> torch.Tensor:
        threshold = 0.5
        binary_img = (image_tensor > threshold).float()

        return binary_img

def create_dataloaders_MovingMNIST(batchsize: int = 16,
                                   num_workers: int = 8):
        
    transforms = v2.Compose([
        v2.ToDtype(dtype= torch.uint8,
                   scale= True),
        v2.Lambda(binarize_image)
    ])

    train_dataset = torchvision.datasets.MovingMNIST(root= "../data/MovingMNIST",
                                                     split= "train",
                                                     download= True,
                                                     transform= transforms)

    test_dataset = torchvision.datasets.MovingMNIST(root= "../data/MovingMNIST",
                                                     split= "test",
                                                     download= True,
                                                     transform= transforms)

    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size= batchsize,
                                  num_workers= num_workers,
                                  shuffle= True)
    
    test_dataloader = DataLoader(dataset= test_dataset,
                                  batch_size= batchsize,
                                  num_workers= num_workers,
                                  shuffle= True)

    return train_dataloader, test_dataloader, train_dataset, test_dataset
