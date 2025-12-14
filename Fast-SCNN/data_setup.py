"""
Containing various classes and functions for creating and loading dataloaders for segmentation datasets.
"""
import torch
import torchvision
import torchvision.transforms.v2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict
from torch.utils.data import random_split

class ISBIDataset(Dataset):

    """
    A torch.utils.data.Dataset subclass for creating train, test, and validation datasets from the ISBI data root directory.
    """
    def __init__(self,
                 data_dir: str,
                 transforms: torchvision.transforms = None):
        
        self.data_dir = data_dir
        self.transforms = transforms
        
        self.img_paths_list = sorted(list(Path(data_dir).glob("imgs/*.png")))
        self.label_paths_list = sorted(list(Path(data_dir).glob("labels/*.png")))
        self.mask_crop = torchvision.transforms.CenterCrop(size=(324, 836))
    
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
            return self.transforms(img), self.mask_crop(self.transforms(mask)/255)
        else:
            return img, mask



def create_dataloader_ISBI(train_dir: str,
                      test_dir: str,
                      transforms: torchvision.transforms,
                      num_workers: int = 4,
                      batch_size: int = 1,
                      test_val_split: float = 0.5
                      ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:

    """
    Create train, test, and validation dataloaders for the ISBI data using the ISBIDataset class.

    Args:
    train_dir: A string containing the path to the training data.
    test_dir: A string containing the path to the testing data.
    transforms: PyTorch vision transforms to apply on each image.
    num_workers: An integer containing the number of cpu cores to use to load data.
    batch_size: An integer containing the batch size to use.
    test_val_split: A float containing the ratio of test data to use for validation.

    Returns:
    train_dataloader: A dataloader object to train the model on.
    test_dataloader: A dataloader object to test the model on.
    val_dataloader: A dataloader object to validate the model on.
    train_dataset: A train dataset to create train dataloaders or to visualise and explore data.
    test_dataset: A train dataset to create test dataloaders or to visualise and explore data.
    val_dataset: A train dataset to create val dataloaders or to visualise and explore data

    Example usage:
    train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloader_ISBI(train_dir=data_path / "train",
                                                                      test_dir=data_path / "test",
                                                                      transforms=transforms,
                                                                      batch_size=batch_size,
                                                                      num_workers=num_workers,
                                                                     test_val_split=0.5)
    """

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

    """
    A torch.utils.data.Dataset subclass for creating train, test, and validation datasets from the Cityscape data root directory.
    """
    
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                img_transforms: torchvision.transforms.v2.Compose = None,
                sample_size: float = 0.2,
                remove_classes: List[int] = None,
                remap_classes: Dict[int, int] = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transforms = img_transforms
        self.remove_classes = remove_classes
        self.remap_classes = remap_classes
        self.ignore_index = -1

        self.normalize = transforms.v2.Normalize(mean=[0.2868, 0.3251, 0.2838],
                                              std=[0.1760, 0.1808, 0.1775])
        self.mask_resize = transforms.v2.Resize(size=(128, 256),
                                            interpolation=transforms.InterpolationMode.NEAREST)
        
        self.img_list = sorted(list(Path(img_dir).glob("*/*.png")))
        self.mask_list = sorted(list(Path(mask_dir).glob("*/*labelIds.png")))
        self.sample_size = int(sample_size*len(self.img_list))
        self.img_list, self.mask_list = self.img_list[:self.sample_size], self.mask_list[:self.sample_size]

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
        
        img = self.load_images(index=index)
        mask = self.load_masks(index=index)
        
        if self.img_transforms:
            img, mask = self.img_transforms(img, mask)
            mask = (mask* 255).round().long()

        img = self.normalize(img)
        mask = self.mask_resize(mask)
        
        if self.remove_classes:
            for cls in self.remove_classes:
                mask[mask == cls] = self.ignore_index

        if self.remap_classes is not None:
            mask_remapped = torch.zeros_like(mask)
            for orig_class, new_class in self.remap_classes.items():
                mask_remapped[mask == orig_class] = new_class

            mask = mask_remapped

        return img, mask


def create_dataloaders_CS(img_dir: str,
                       mask_dir: str,
                       img_transforms: torchvision.transforms.Compose,
                      batch_size: int,
                      num_workers: int = 4,
                      sample_size: float = 0.2,
                         remove_classes: List[int] = None,
                         remap_classes: Dict[int, int] = None):

    """
    Create train, test, and validation dataloaders for the Cityscape data using the CityscapeDataset class.

    Args:
    train_dir: A string containing the path to the training data.
    test_dir: A string containing the path to the testing data.
    transforms: PyTorch vision transforms to apply on each image.
    num_workers: An integer containing the number of cpu cores to use to load data.
    batch_size: An integer containing the batch size to use.
    sample_size: A float containing the ratio of data to use from the full length of the data.

    Returns:
    train_dataloader: A dataloader object to train the model on.
    test_dataloader: A dataloader object to test the model on.
    val_dataloader: A dataloader object to validate the model on.
    train_dataset: A train dataset to create train dataloaders or to visualise and explore data.
    test_dataset: A train dataset to create test dataloaders or to visualise and explore data.
    val_dataset: A train dataset to create val dataloaders or to visualise and explore data

    Example usage:
    train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloader_ISBI(train_dir=data_path / "train",
                                                                      test_dir=data_path / "test",
                                                                      transforms=transforms,
                                                                      batch_size=batch_size,
                                                                      num_workers=num_workers,
                                                                     sample_size=0.5)
    """

    datasets = {}
    dataloaders = {}
    for item in list(img_dir.iterdir())[::-1]:
        dataset = f"{item}_dataset"
        datasets[dataset] = CityscapeDataset(img_dir=item,
                                          mask_dir=mask_dir / item.stem,
                                          img_transforms=img_transforms,
                                          sample_size=sample_size,
                                            remove_classes= remove_classes,
                                            remap_classes = remap_classes)
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

    """
    A torch.utils.data.Dataset subclass for creating train, test, and validation datasets from the Carvana data root directory.
    """
    
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 transform: torchvision.transforms,
                 sample_size: float = 0.2):
        
        self.transform = transform
        self.img_list = sorted(list(img_dir.glob("*.jpg")))
        self.mask_list = sorted(list(mask_dir.glob("*.png")))
        self.mask_resize = torchvision.transforms.Resize(size=(324, 836))
        
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

        if self.mask_resize:
            mask = self.mask_resize(mask)

        return img, mask

def create_dataloaders_Carvana(img_dir: str,
                               mask_dir: str,
                               transform: torchvision.transforms,
                               mask_resize: Tuple[int, int] = None,
                               sample_size: float = 0.2,
                               batch_size: int = 2,
                               num_workers: int = 4,
                               train_test_val_split: Tuple[float, float, float] = (0.7, 0.2, 0.1)
                              ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:

    """
    Create train, test, and validation dataloaders for the Carvana data using the CarvanaDataset class.

    Args:
    train_dir: A string containing the path to the training data.
    test_dir: A string containing the path to the testing data.
    transforms: PyTorch vision transforms to apply on each image.
    num_workers: An integer containing the number of cpu cores to use to load data.
    batch_size: An integer containing the batch size to use.
    sample_size: A float containing the ratio of data to use from the full length of the data.
    crop_size: A tuple of ints containing the shape after cropping the mask outputs to fit the UNet output shape.
    
    Returns:
    train_dataloader: A dataloader object to train the model on.
    test_dataloader: A dataloader object to test the model on.
    val_dataloader: A dataloader object to validate the model on.
    train_dataset: A train dataset to create train dataloaders or to visualise and explore data.
    test_dataset: A train dataset to create test dataloaders or to visualise and explore data.
    val_dataset: A train dataset to create val dataloaders or to visualise and explore data

    Example usage:
    train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloader_ISBI(train_dir=data_path / "train",
                                                                      test_dir=data_path / "test",
                                                                      transforms=transforms,
                                                                      batch_size=batch_size,
                                                                      num_workers=num_workers,
                                                                     sample_size=0.5
                                                                     crop_size= (324, 836))
    """
    
    full_dataset = CarvanaDataset(img_dir= img_dir,
                                  mask_dir= mask_dir,
                                  transform= transform,
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

class ADE20KDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 img_transforms: torchvision.transforms.v2.Compose = None,
                 remove_classes: List[int] = None,
                 remap_classes: Dict[int, int] = None,
                 sample_size: float = 0.2):

        self.img_transforms = img_transforms

        self.mask_resize = transforms.v2.Resize(size = (128, 256),
                                                interpolation = transforms.v2.InterpolationMode.NEAREST)
        self.normalize = transforms.v2.Normalize(mean= [0.5227, 0.4778, 0.4298],
                                                std= [0.2149, 0.2192, 0.2266])

        self.img_list = sorted(list(img_dir.glob("*.jpg")))
        self.mask_list = sorted(list(mask_dir.glob("*.png")))
        self.sample_size = int(sample_size * len(self.img_list))
        assert len(self.img_list) !=0 or len(self.mask_list) != 0, "Image or Mask list is empty, check input directory or file type"
        assert len(self.img_list) == len(self.mask_list), "Number of images and mask must be the same"
        self.img_list = self.img_list[: self.sample_size]
        self.mask_list = self.mask_list[: self.sample_size]

        self.ignore_index = -1
        self.remove_classes = remove_classes
        self.remap_classes = remap_classes
        
    def load_image(self,
                   idx: int):
        
        return Image.open(self.img_list[idx]).convert("RGB")

    def load_mask(self,
                  idx: int):

        return Image.open(self.mask_list[idx])

    def __len__(self):
        
        return len(self.img_list)

    def __getitem__(self,
                    idx: int):

        img = self.load_image(idx= idx)
        mask = self.load_mask(idx= idx)
        
        if self.img_transforms:
            img, mask = self.img_transforms(img, mask)
            mask = (mask * 255).round().long()

        img = self.normalize(img)
        mask = self.mask_resize(mask)

        if self.remove_classes:
            for class_idx in self.remove_classes:
                mask[mask == class_idx] = self.ignore_index

        if self.remap_classes:
            mask_remapped = torch.zeros_like(mask)
            for original_id, new_id in self.remap_classes.items():
                mask_remapped[mask == original_id] = new_id

            mask = mask_remapped
            
        return img, mask

def create_dataloader_ADE(data_path: str,
                          img_transforms: torchvision.transforms.Compose,
                          sample_size: float,
                          batch_size: int,
                          num_workers: int = 4,
                         test_val_split: float = 0.5,
                         remove_classes: List[int] = None,
                         remap_classes: Dict[int,  int] = None):

    train_dataset = ADE20KDataset(img_dir= data_path / "ADEChallengeData2016" / "images" / "training",
                                  mask_dir= data_path / "ADEChallengeData2016" / "annotations" / "training",
                                  sample_size= sample_size,
                                  img_transforms= img_transforms,
                                 remove_classes= remove_classes,
                                 remap_classes= remap_classes)

    test_dataset = ADE20KDataset(img_dir= data_path / "ADEChallengeData2016" / "images" / "validation",
                                  mask_dir= data_path / "ADEChallengeData2016" / "annotations" / "validation",
                                  sample_size= sample_size,
                                  img_transforms= img_transforms,
                                remove_classes= remove_classes,
                                remap_classes= remap_classes)

    test_split = int(test_val_split * len(test_dataset))
    val_split = len(test_dataset) - test_split

    test_dataset, val_dataset = random_split(dataset= test_dataset,
                                                lengths=[test_split, val_split])

    train_dataloader = DataLoader(dataset= train_dataset,
                                    batch_size= batch_size,
                                    num_workers= num_workers,
                                    shuffle= True)

    test_dataloader= DataLoader(dataset= test_dataset,
                                    batch_size=batch_size,
                                    num_workers= num_workers,
                                    shuffle= True)
    val_dataloader = DataLoader(dataset= val_dataset,
                                batch_size= batch_size,
                                num_workers= num_workers,
                                shuffle= False)

    return train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset

def choose_dataloader(data_path: str,
                      dataset_name: str,
                     img_transforms: torchvision.transforms.Compose,
                     batch_size: int,
                     num_workers: int,
                     sample_size: float = 0.1,
                     remove_classes: List[int] = None,
                     remap_classes: Dict[int, int] = None):

    """
    A function to choose create_dataloader based on the data downloaded.

    Args:
    data_path: A string containing the path to the root directory of the downloaded data.
    dataset_name: The name of the downloaded dataset (e.g "Cityscape", "ISBI", "Carvana").
    transforms: A torchvision transform to apply to each dataset.
    batch_size: An integer containing the batch size to use.
    num_workers: An integer containing the number of cpu cores to use for dataloading.
    sample_size: A float containing the ratio of data to use from the full length of the data.

    Returns:
    train_dataloader: A dataloader object to train the model on.
    test_dataloader: A dataloader object to test the model on.
    val_dataloader: A dataloader object to validate the model on.
    train_dataset: A train dataset to create train dataloaders or to visualise and explore data.
    test_dataset: A train dataset to create test dataloaders or to visualise and explore data.
    val_dataset: A train dataset to create val dataloaders or to visualise and explore data

    train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = choose_dataloader(data_path= data_path,
                                                                                                                dataset_name=args.dataset_name,
                                                                                                               transforms=transforms,
                                                                                                               batch_size= args.batch_size,
                                                                                                               num_workers = args.num_workers,
                                                                                                               sample_size = args.sample_size)
    
    """
    
    if dataset_name == "ISBI":
        train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloader_ISBI(train_dir=data_path / "train",
                                                                      test_dir=data_path / "test",
                                                                      transforms=img_transforms,
                                                                      batch_size=batch_size,
                                                                      num_workers=num_workers,
                                                                     test_val_split=0.5)
    elif dataset_name == "Cityscape":
        (test_dataloader, train_dataloader, val_dataloader), (test_dataset, train_dataset, val_dataset) = create_dataloaders_CS(img_dir= data_path / "Cityscape Dataset" / "leftImg8bit",
                                                                                                                       mask_dir= data_path / "Fine Annotations" / "gtFine",
                                                                                                                       img_transforms=img_transforms,
                                                                                                                          batch_size=batch_size,
                                                                                                                       sample_size=sample_size,
                                                                                                                               num_workers=num_workers,
                                                                                                                               remove_classes = remove_classes,
                                                                                                                               remap_classes = remap_classes)
    elif dataset_name == "Carvana":
        train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloaders_Carvana(img_dir=data_path / "train_images",
                                                                                                                         mask_dir=data_path / "train_masks",
                                                                                                                         transform= img_transforms,
                                                                                                                         sample_size = sample_size,
                                                                                                                         batch_size=batch_size,
                                                                                                                         num_workers=num_workers,
                                                                                                                        train_test_val_split=(0.7, 0.2, 0.1))

    else:
        train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset = create_dataloader_ADE(data_path= data_path,
                                                                                                                    img_transforms= img_transforms,
                                                                                                                    sample_size= sample_size,
                                                                                                                    batch_size= batch_size,
                                                                                                                    num_workers= num_workers,
                                                                                                                    test_val_split= 0.5,
                                                                                                                   remove_classes= remove_classes,
                                                                                                                   remap_classes= remap_classes
                                                                                                                             )
         
    return train_dataloader, test_dataloader, val_dataloader, train_dataset, test_dataset, val_dataset

    



