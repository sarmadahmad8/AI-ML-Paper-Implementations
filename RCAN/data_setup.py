import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
import random

class Div2K(Dataset):
    def __init__(self,
                 image_dir: str,
                 scale: int,
                 sample_size: float,
                 crops_per_image: int = 5):

        self.img_list = sorted(list(image_dir.glob("*/*.png")))
        self.sample_size = int(sample_size * len(self.img_list))
        self.img_list = self.img_list[:self.sample_size]
        self.scale = scale
        self.patch_size_hr = scale * 48
        self.crops_per_image = crops_per_image
        
        assert self.img_list != 0, "image directory has no images, check directory"

    def load_image(self,
                   index):

        img_index = index // self.crops_per_image

        hr_img = Image.open(self.img_list[index]).convert("RGB")
        return hr_img

    def _transforms(self,
                   hr_img: Image.Image):

        width, height = hr_img.size
        patch_size_lr = 48

        angles = [0, 90, 180, 270]
        angle = random.choice(angles)

        if angle != 0:
            hr_img = F.rotate(hr_img,
                              angle= angle)

        if random.random() > 0.5:
            hr_img = F.hflip(hr_img)

        left = random.randint(0, width - self.patch_size_hr)
        top = random.randint(0, height - self.patch_size_hr)

        hr_img = F.crop(img= hr_img,
                       top= top,
                       left= left,
                       height= self.patch_size_hr,
                       width= self.patch_size_hr)

        lr_img = F.resize(img= hr_img,
                          size=(patch_size_lr, patch_size_lr),
                          interpolation= v2.InterpolationMode.BICUBIC)

        lr_img, hr_img = F.to_tensor(pic= lr_img), F.to_tensor(pic= hr_img)

        

        return lr_img, hr_img
        

    def __len__(self):
        
        return len(self.img_list) * self.crops_per_image

    def __getitem__(self,
                    index: int):
        
        img_index = index // self.crops_per_image

        hr_img = self.load_image(index= img_index)

        lr_img, hr_img = self._transforms(hr_img)

        return lr_img, hr_img

def create_dataloaders(img_dir: str,
                       scale: int,
                       sample_size: float,
                       batch_size: int,
                       crops_per_image: int = 5,
                       test_val_split: float = 0.5,
                       num_workers: int = 4):

    train_dataset = Div2K(image_dir= img_dir / "DIV2K_train_HR",
                          scale= scale,
                          sample_size= sample_size,
                          crops_per_image= crops_per_image)

    test_dataset = Div2K(image_dir= img_dir / "DIV2K_valid_HR",
                         scale= scale,
                         sample_size= sample_size,
                         crops_per_image= crops_per_image)

    test_split = int(test_val_split * len(test_dataset))
    val_split = len(test_dataset) - test_split

    test_dataset, val_dataset = random_split(dataset= test_dataset,
                                             lengths= [test_split, val_split])

    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size= batch_size,
                                  num_workers= num_workers,
                                  shuffle= True)

    test_dataloader = DataLoader(dataset= test_dataset,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 shuffle= True)

    val_dataloader = DataLoader(dataset= val_dataset,
                                batch_size= batch_size,
                                num_workers= num_workers,
                                shuffle= False)

    return train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset
                       
