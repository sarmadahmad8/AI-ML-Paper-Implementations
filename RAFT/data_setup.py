import torch
from torchvision.datasets import KittiFlow, Sintel, Middlebury2014Stereo
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, Subset

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
    
    train_length = int(len(dataset) * train_test_split)
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset=dataset,
                                               lengths=[train_length, test_length])
    
    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size = batch_size,
                                  num_workers= num_workers,
                                  shuffle = True)
    
    test_dataloader = DataLoader(dataset= test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= False)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def create_dataloaders_Sintel(batch_size: int,
                              num_workers: int,
                              train_test_split: float = 0.9,
                              sample_size: float = 1.0):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype= torch.float32,
                   scale= True)
    ])
    
    dataset = Sintel(root= "../data", 
                     split= "train",
                     transforms = transforms)
    
    train_length = int(len(dataset) * train_test_split)
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset=dataset,
                                               lengths=[train_length, test_length])

    train_indices = torch.arange(int(len(train_dataset) * sample_size))
    test_indices = torch.arange(int(len(test_dataset) * sample_size))

    train_dataset, test_dataset = Subset(train_dataset, train_indices), Subset(test_dataset, test_indices)
    
    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size = batch_size,
                                  num_workers= num_workers,
                                  shuffle = True)
    
    test_dataloader = DataLoader(dataset= test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= False)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def create_dataloaders_Middlebury(batch_size: int,
                                 num_workers: int):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype= torch.float32,
                   scale= True)
    ])
    
    train_dataset = Middlebury2014Stereo(root= "../data", 
                                         split= "train",
                                         transforms = transforms,
                                         download= True)
    
    train_length = int(len(dataset) * train_test_split)
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset=dataset,
                                               lengths=[train_length, test_length])
    
    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size = batch_size,
                                  num_workers= num_workers,
                                  shuffle = True)
    
    test_dataloader = DataLoader(dataset= test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= False)

    return train_dataloader, test_dataloader, train_dataset, test_dataset
