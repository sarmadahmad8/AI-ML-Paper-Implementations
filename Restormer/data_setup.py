import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
import random
from typing import List

class GoPro(Dataset):

    def __init__(self,
                 img_dir: str,
                 crops_per_image: int,
                 patch_size: int,
                 sample_size: float,
                 augment: bool):

        self.blur_img_list = sorted(list(img_dir.glob("*/blur/*/*.png")))
        self.sharp_img_list = sorted(list(img_dir.glob("*/sharp/*/*.png")))
        self.sample_size = int(sample_size * len(self.sharp_img_list))
        self.crops_per_image = crops_per_image
        self.augment = augment
        self.patch_size = patch_size
        
        self.blur_img_list = self.blur_img_list[: self.sample_size]
        self.sharp_img_list = self.sharp_img_list[: self.sample_size]

        assert len(self.blur_img_list) != 0 or len(self.sharp_img_list) != 0, "Image directory empty, check given data directory"

    def load_image_pairs(self,
                         index: int):
        
        img_index = index // self.crops_per_image

        blur_img = Image.open(self.blur_img_list[img_index]).convert(mode= "RGB")
        sharp_img = Image.open(self.sharp_img_list[img_index]).convert(mode= "RGB")

        return blur_img, sharp_img

    def _augment(self,
                 blur_img: Image.Image,
                 sharp_img: Image.Image):

        
        if random.random() > 0.6:
            blur_img = F.hflip(img= blur_img)
            sharp_img = F.hflip(img= sharp_img)

            blur_img = F.vflip(img= blur_img)
            sharp_img = F.vflip(img= sharp_img)

        return blur_img, sharp_img

    def _transforms(self,
                    blur_img: Image.Image,
                    sharp_img: Image.Image):

        height, width = blur_img.size

        left = random.randint(0, width - self.patch_size)
        top = random.randint(0, height - self.patch_size)

        blur_img = F.crop(img= blur_img,
                          top= top,
                          left= left,
                          height= self.patch_size,
                          width= self.patch_size)
        
        sharp_img = F.crop(img= sharp_img,
                               top= top,
                               left= left,
                               height= self.patch_size,
                               width= self.patch_size)

        blur_img, sharp_img = F.to_tensor(blur_img), F.to_tensor(sharp_img)

        return blur_img, sharp_img

    def __len__(self):

        return len(self.sharp_img_list) * self.crops_per_image

    def __getitem__(self,
                    index: int):

        blur_img, sharp_img = self.load_image_pairs(index= index)

        if self.augment == True:
            
            blur_img, sharp_img = self._augment(blur_img= blur_img,
                                                sharp_img= sharp_img)

        blur_img, sharp_img = self._transforms(blur_img= blur_img,
                                               sharp_img= sharp_img)

        return blur_img, sharp_img

def create_dataloaders_GoPro(img_dir: str,
                             crops_per_image: int,
                             patch_size: int,
                             sample_size: float,
                             augment: bool,
                             batch_size: int,
                             train_test_val_split: List[float],
                             num_workers: int = 4):

    dataset = GoPro(img_dir= img_dir,
                    crops_per_image= crops_per_image,
                    patch_size= patch_size,
                    sample_size= sample_size,
                    augment= augment)

    train_length = int(train_test_val_split[0] * len(dataset))
    test_length = int(train_test_val_split[1] * len(dataset))
    val_length = len(dataset) - train_length - test_length

    train_dataset, test_dataset, val_dataset = random_split(dataset= dataset,
                                                            lengths= [train_length, test_length, val_length])

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

    
class Urban100(Dataset):

    def __init__(self,
                 img_dir: str,
                 crops_per_image: int,
                 scale: int,
                 patch_size_lr: int,
                 sample_size: float,
                 augment: bool = False,
                 add_noise: bool = False,
                 sigma: float = 0.1):

        if add_noise:
            scale = 2
        self.img_dir = img_dir / "Urban 100" / f"X{scale} Urban100" / f"X{scale}"
        self.hr_img_list = sorted(list(self.img_dir.glob("*/*_HR.png")))
        self.lr_img_list = sorted(list(self.img_dir.glob("*/*_LR.png")))
        self.sample_size = int(sample_size * len(self.hr_img_list))
        self.hr_img_list = self.hr_img_list[: self.sample_size]
        self.lr_img_list = self.lr_img_list[: self.sample_size]
        self.scale = scale
        self.crops_per_image = crops_per_image
        self.patch_size_lr = patch_size_lr
        self.patch_size_hr = scale * patch_size_lr
        self.augment = augment
        self.add_noise = add_noise
        self.sigma = sigma

        assert len(self.hr_img_list) != 0 or len(self.lr_img_list) != 0, "Image directory does not have any images, check directory."

    def load_hr_image(self,
                      index: int):

        hr_img = Image.open(self.hr_img_list[index]).convert(mode="RGB")

        return hr_img

    def load_lr_image(self,
                      index: int):

        lr_img = Image.open(self.lr_img_list[index]).convert(mode="RGB")

        return lr_img

    def __len__(self):

        return len(self.hr_img_list) * self.crops_per_image

    def __getitem__(self,
                    index: int):

        img_index = index // self.crops_per_image

        hr_img = self.load_hr_image(img_index)
        lr_img = self.load_lr_image(img_index)

        if self.augment:
            hr_img, lr_img = self._augment(hr_img, lr_img)

        hr_img, lr_img = self._transforms(hr_img,
                                          lr_img)

        return lr_img, hr_img

    def _augment(self,
                 hr_img: Image.Image,
                 lr_img: Image.Image):

        hr_width, hr_height = hr_img.size
        lr_width, lr_height = lr_img.size

        angles = [0, 90, 180, 270]
        angle = random.choice(angles)

        if angle != 0:
            hr_img = F.rotate(img= hr_img,
                              angle= angle)
            lr_img = F.rotate(img= lr_img,
                              angle = angle)

        if random.random() > 0.5:

            hr_img = F.hflip(img= hr_img)
            lr_img = F.hflip(img= lr_img)

        return hr_img, lr_img

    def _transforms(self,
                    hr_img: Image.Image,
                    lr_img: Image.Image):

        hr_width, hr_height = hr_img.size
        lr_width, lr_height = lr_img.size

        left = random.randint(0, hr_width - self.patch_size_hr)
        top = random.randint(0, hr_height - self.patch_size_hr)

        hr_img = F.crop(img= hr_img,
                        left= left,
                        top= top,
                        height= self.patch_size_hr,
                        width= self.patch_size_hr)

        lr_img = F.crop(img= lr_img,
                        left= left // self.scale,
                        top= top // self.scale,
                        height= self.patch_size_hr // self.scale,
                        width= self.patch_size_hr // self.scale)

        hr_img, lr_img = F.to_tensor(pic= hr_img), F.to_tensor(pic= lr_img)

        if self.add_noise:
    
            noise = torch.randn_like(hr_img) * self.sigma
            noisy_img = lr_img + noise
            noisy_img = noisy_img.clamp(max=1.0, min=0.0)

            return hr_img, noisy_img

        return hr_img, lr_img

def create_dataloaders_Urban100(img_dir: str,
                                crops_per_image: int,
                                scale: int,
                                sample_size: float,
                                batch_size: int,
                                patch_size_lr: int = 64,
                                train_test_val_split: List[float] = None,
                                num_workers: int = 4,
                                add_noise: bool = False,
                                sigma: float = 0.1):
    if train_test_val_split:
        augment = True
    else:
        augment = False

    dataset = Urban100(img_dir= img_dir,
                       scale= scale,
                       crops_per_image= crops_per_image,
                       sample_size= sample_size,
                       patch_size_lr = patch_size_lr,
                       augment = augment,
                       add_noise = add_noise,
                       sigma = sigma)

    if train_test_val_split:
        
        train_split = int(train_test_val_split[0] * len(dataset))
        test_split = int(train_test_val_split[1] * len(dataset))
        val_split = len(dataset) - train_split - test_split

        train_dataset, test_dataset, val_dataset = random_split(dataset= dataset,
                                                                lengths= [train_split, test_split, val_split])

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

    else:
        
        val_dataloader = DataLoader(dataset= dataset,
                                    batch_size= batch_size,
                                    num_workers= num_workers,
                                    shuffle= False)
        
        return val_dataloader, dataset

class Kodak24(Dataset):
    def __init__(self,
                 image_dir: str,
                 scale: int,
                 sample_size: float,
                 patch_size_lr: int = 64,
                 crops_per_image: int = 5,
                 augment: bool = False,
                 add_noise: bool = True,
                 sigma: float = 0.1):

        self.img_list = sorted(list(image_dir.glob("*/*/*.png")))
        self.sample_size = int(sample_size * len(self.img_list))
        self.img_list = self.img_list[:self.sample_size]
        self.scale = scale
        self.patch_size_lr = patch_size_lr
        self.patch_size_hr = scale * patch_size_lr
        self.crops_per_image = crops_per_image
        self.augment = augment
        self.add_noise = add_noise
        self.sigma = sigma
        
        assert self.img_list != 0, "image directory has no images, check directory"

    def load_image(self,
                   index):

        img_index = index // self.crops_per_image

        hr_img = Image.open(self.img_list[index]).convert("RGB")
        return hr_img

    def _augment(self,
                 hr_img: Image.Image):

        width, height = hr_img.size
        patch_size_lr = self.patch_size_lr

        angles = [0, 90, 180, 270]
        angle = random.choice(angles)

        if angle != 0:
            hr_img = F.rotate(hr_img,
                              angle= angle)

        if random.random() > 0.5:
            hr_img = F.hflip(hr_img)

        return hr_img

    def _transforms(self,
                   hr_img: Image.Image):

        width, height = hr_img.size
        patch_size_lr = self.patch_size_lr

        left = random.randint(0, width - self.patch_size_hr)
        top = random.randint(0, height - self.patch_size_hr)

        hr_img = F.crop(img= hr_img,
                       top= top,
                       left= left,
                       height= self.patch_size_hr,
                       width= self.patch_size_hr)

        if self.add_noise == False:
            lr_img = F.resize(img= hr_img,
                              size=(patch_size_lr, patch_size_lr),
                              interpolation= v2.InterpolationMode.BICUBIC)

            
            lr_img, hr_img = F.to_tensor(pic= lr_img), F.to_tensor(pic= hr_img)

            return lr_img, hr_img
            
        else:
            hr_img = F.to_tensor(pic= hr_img)
            
            noise = torch.randn_like(hr_img) * self.sigma
            noise_img = hr_img + noise
            noise_img = noise_img.clamp(max = 1.0, min = 0.0)

            return noise_img, hr_img

    def __len__(self):
        
        return len(self.img_list) * self.crops_per_image

    def __getitem__(self,
                    index: int):
        
        img_index = index // self.crops_per_image

        hr_img = self.load_image(index= img_index)

        if self.augment:
            hr_img = self._augment(hr_img)

        lr_img, hr_img = self._transforms(hr_img)

        return lr_img, hr_img

def create_dataloaders_Kodak24(img_dir: str,
                               scale: int,
                               sample_size: float,
                               batch_size: int,
                               crops_per_image: int = 5,
                               patch_size_lr: int = 64,
                               num_workers: int = 4,
                               add_noise: bool = True,
                               sigma: float = 0.1):

    val_dataset = Kodak24(image_dir= img_dir,
                          scale= scale,
                          sample_size= sample_size,
                          crops_per_image= crops_per_image,
                          patch_size_lr = patch_size_lr,
                          augment = False,
                          add_noise = add_noise,
                          sigma = sigma)


    val_dataloader = DataLoader(dataset= val_dataset,
                                  batch_size= batch_size,
                                  num_workers= num_workers,
                                  shuffle= False)

    return val_dataloader, val_dataset
