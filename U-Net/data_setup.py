
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Tuple
from torch.utils.data import random_split

class ISBIDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transforms: torchvision.transforms = None):
        
        self.data_dir = data_dir
        self.transforms = transforms
        
        self.img_paths_list = sorted(list(Path(data_dir).glob("imgs/*.png")))
        self.label_paths_list = sorted(list(Path(data_dir).glob("labels/*.png")))
        self.mask_crop = torchvision.transforms.CenterCrop(size=(324, 324))
    
    def load_images(self, index: int) -> Image.Image:
        img = Image.open(self.img_paths_list[index])
        return img

    def load_masks(self, index: int) -> Image.Image:
        mask = Image.open(self.label_paths_list[index])
        return mask

    def __len__(self) -> int:
        return len(self.img_paths_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.load_images(index= index)
        mask = self.load_masks(index= index)
        if self.transforms:
            return self.transforms(img), self.mask_crop(self.transforms(mask))
        else:
            return img, mask



def create_dataloader_ISBI(train_dir: str,
                      test_dir: str,
                      transforms: torchvision.transforms,
                      num_workers: int = 4,
                      batch_size: int = 1,
                      test_val_split: float = 0.5
                      ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:


    train_dataset = ISBIDataset(data_dir=train_dir,
                                transforms=transforms)
    test_dataset = ISBIDataset(data_dir=test_dir,
                               transforms=transforms)

    test_split = int(test_val_split * len(test_dataset))
    val_split = int(len(test_dataset) - test_split)

    test_dataset, val_dataset = random_split(dataset= test_dataset, 
                                             lengths=[test_split, val_split])

    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset


class CityscapeDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                transform: torchvision.transforms = None,
                sample_size: float = 0.2):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.img_list = sorted(list(Path(img_dir).glob("*/*.png")))
        self.mask_list = sorted(list(Path(mask_dir).glob("*/*labelIds.png")))
        self.sample_size = int(sample_size*len(self.img_list))
        self.img_list, self.mask_list = self.img_list[:self.sample_size], self.mask_list[:self.sample_size]

        self.mask_crop = torchvision.transforms.CenterCrop(size=(324, 836))
        # print(len(self.img_list), len(self.mask_list))
        assert len(self.img_list)==len(self.mask_list), "Number of items in mask directory do not match number of items in image directory"
    def load_images(self, 
                    index: int) -> Image.Image:
        return Image.open(self.img_list[index])

    def load_masks(self,
                   index: int) -> Image.Image:
        return Image.open(self.mask_list[index])

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, 
                    index: int) -> Tuple[torch.tensor, torch.tensor]:
        if self.transform:
            return self.transform(self.load_images(index=index)), self.mask_crop(self.transform(self.load_masks(index=index)))

        else:
            return self.load_images(index), self.load_masks(index)


def create_dataloaders_CS(img_dir: str,
                       mask_dir: str,
                       transform: torchvision.transforms,
                      batch_size: int,
                      num_workers: int = 4,
                      sample_size: float = 0.2):

    datasets = {}
    dataloaders = {}
    for item in list(img_dir.iterdir())[::-1]:
        dataset = f"{item}_dataset"
        datasets[dataset] = CityscapeDataset(img_dir=item,
                                          mask_dir=mask_dir / item.stem,
                                          transform=transform,
                                          sample_size=sample_size)
        dataloader = f"{item}_dataloader"
        if item.stem != "val":
            dataloaders[dataloader] = DataLoader(dataset=datasets[dataset],
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True)
        else:
            dataloaders[dataloader] = DataLoader(dataset=datasets[dataset],
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=False)
            
    
    return tuple(dataloaders.values()), tuple(datasets.values())

class CarvanaDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 transform: torchvision.transforms,
                 cropsize: Tuple[int, int] = None,
                 sample_size: float = 0.2):
        
        self.transform = transform
        self.img_list = sorted(list(img_dir.glob("*.jpg")))
        self.mask_list = sorted(list(mask_dir.glob("*.png")))
        if cropsize:
            self.center_crop = torchvision.transforms.CenterCrop(size=(cropsize))
        else: 
            self.center_crop = None
        self.sample_size = int(sample_size*len(self.img_list))
        self.mask_list, self.img_list = self.mask_list[: self.sample_size], self.img_list[:self.sample_size]

        assert len(self.img_list)==len(self.mask_list), "Number of images does not match the number of masks"
        
    def load_img(self,
                 index: int):
        return Image.open(self.img_list[index])

    def load_mask(self, 
                  index: int):
        return Image.open(self.mask_list[index])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,
                   index: int):

        img = self.load_img(index=index)
        mask = self.load_mask(index=index)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        if self.center_crop:
            mask = self.center_crop(mask)

        return img, mask

def create_dataloaders_Carvana(img_dir: str,
                               mask_dir: str,
                               transform: torchvision.transforms,
                               cropsize: Tuple[int, int] = None,
                               sample_size: float = 0.2,
                               batch_size: int = 2,
                               num_workers: int = 4,
                               train_test_val_split: Tuple[float, float, float] = (0.7, 0.2, 0.1)
                              ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:

    full_dataset = CarvanaDataset(img_dir= img_dir,
                                  mask_dir= mask_dir,
                                  transform= transform,
                                  cropsize= cropsize,
                                  sample_size= sample_size)

    train_split = int(train_test_val_split[0]*len(full_dataset))
    test_split = int(train_test_val_split[1]*len(full_dataset))
    val_split = len(full_dataset) - train_split - test_split
    
    train_dataset, test_dataset, val_dataset = random_split(dataset=full_dataset,
                                                            lengths=[train_split, test_split, val_split]
                                                           )

    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset


