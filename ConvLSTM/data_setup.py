import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torchvision
from torchvision.transforms import v2

def binarize_image(image_tensor: torch.Tensor) -> torch.Tensor:
        threshold = 0.5
        binary_img = (image_tensor > threshold).float()

        return binary_img

def create_dataloaders_MovingMNIST(batchsize: int = 16,
                                   num_workers: int = 8):
        
    transforms = v2.Compose([
        v2.ToDtype(dtype= torch.float32,
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

    dataset = ConcatDataset(datasets=[train_dataset, test_dataset])

    train_length = int(0.8 * len(dataset))
    test_length = int(0.1 * len(dataset))
    val_length = len(dataset) - train_length - test_length

    train_dataset, test_dataset, val_dataset = random_split(dataset= dataset,
                                                            lengths= [train_length, test_length, val_length])

    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size= batchsize,
                                  num_workers= num_workers,
                                  shuffle= True)
    
    test_dataloader = DataLoader(dataset= test_dataset,
                                  batch_size= batchsize,
                                  num_workers= num_workers,
                                  shuffle= True)

    val_dataloader = DataLoader(dataset= val_dataset,
                                batch_size= batchsize,
                                num_workers= num_workers,
                                shuffle= False)

    return train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset
