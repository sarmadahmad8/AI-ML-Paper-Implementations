import torch
import torchvision
from torchvision.transforms import v2
from torchvision.datasets import Kitti2015Stereo, Kitti2012Stereo, SintelStereo
from torch.utils.data import DataLoader, random_split

def create_dataloaders_kitti2012(batch_size: int,
                                num_workers: int = 8,
                                train_test_split: float = 0.9):

    transforms = v2.Compose([v2.CenterCrop(size=(368, 1232)),
                             v2.ToImage(),
                             v2.ToDtype(dtype=torch.float32, scale=True)])

    dataset = Kitti2012Stereo(root = "../data/",
                              split = "train",
                              transforms= transforms)

    train_length = int(len(dataset) * train_test_split)
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset=dataset,
                                               lengths=[train_length, test_length])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size= batch_size,
                                  num_workers= num_workers,
                                  shuffle= True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= False)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def create_dataloaders_kitti2015(batch_size: int,
                                num_workers: int = 8,
                                train_test_split: float = 0.9):

    transforms = v2.Compose([v2.ToImage(),
                             v2.ToDtype(dtype=torch.float32, scale=True)])

    dataset = Kitti2015Stereo(root = "../data/",
                              split = "train",
                              transforms= transforms)

    train_length = int(len(dataset) * train_test_split)
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset=dataset,
                                               lengths=[train_length, test_length])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size= batch_size,
                                  num_workers= num_workers,
                                  shuffle= True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= False)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

def create_dataloaders_SintelStereo(batch_size: int,
                                    num_workers: int = 8,
                                    train_test_split: float = 0.9):

    transforms = v2.Compose([v2.ToImage(),
                             v2.ToDtype(dtype=torch.float32, scale=True)])

    dataset = SintelStereo(root = "../data/",
                              transforms= transforms)

    train_length = int(len(dataset) * train_test_split)
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset=dataset,
                                               lengths=[train_length, test_length])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size= batch_size,
                                  num_workers= num_workers,
                                  shuffle= True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= False)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders_SintelStereo(batch_size=1)

len(train_dataloader), len(test_dataloader), len(train_dataset), len(test_dataset)
