import torch
from torchvision.datasets import KittiFlow, Sintel
from torchvision.transforms import v2
from torch.utils.data import DataLoader

def create_dataloaders_KittiFlow(batch_size: int,
                                 num_workers: int):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype= torch.float32,
                   scale= True)
    ])
    
    train_dataset = KittiFlow(root= "../data", 
                              split= "train",
                              transforms = transforms)
    
    test_dataset = KittiFlow(root= "../data",
                             split= "test",
                             transforms= transforms)
    
    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size = batch_size,
                                  num_workers= num_workers,
                                  shuffle = True)
    
    test_dataloader = DataLoader(dataset= test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= True)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def create_dataloaders_Sintel(batch_size: int,
                              num_workers: int):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype= torch.float32,
                   scale= True)
    ])
    
    train_dataset = Sintel(root= "../data", 
                           split= "train",
                           transforms = transforms)
    
    test_dataset = Sintel(root= "../data",
                          split= "test",
                          transforms= transforms)
    
    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size = batch_size,
                                  num_workers= num_workers,
                                  shuffle = True)
    
    test_dataloader = DataLoader(dataset= test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= True)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders_Sintel(batch_size= 4,
                                                                                              num_workers= 8)

len(train_dataloader), len(test_dataloader), len(train_dataset), len(test_dataset)
