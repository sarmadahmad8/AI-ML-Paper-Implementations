from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas
import torchvision.transforms.v2

class CelebA(Dataset):
    def __init__(self,
                 img_path: str,
                 partition: set,
                 img_transforms: torchvision.transforms.v2.Compose,
                 sample_size: float = 1.0):

        self.total_img_list = list(img_path.glob("*/*.jpg"))
        self.sample_size = int(sample_size * len(self.total_img_list))
        self.total_img_list = self.total_img_list[:self.sample_size]
        self.split_img_list = sorted([path for path in self.total_img_list if path.name in partition])
        self.img_transforms = img_transforms
        
    def load_image(self,
                   index: int):
        return Image.open(self.split_img_list[index]).convert("RGB")

    def __len__(self):
        return len(self.split_img_list)

    def __getitem__(self,
                    index: int):
        img = self.load_image(index= index)
        
        if self.img_transforms:
            img = self.img_transforms(img)

        return img, img


def create_dataloaders(data_path: str,
                       img_transforms: torchvision.transforms.v2.Compose,
                      batch_size: int,
                      num_workers: int = 4,
                      sample_size: float = 1.0):

    split_df = pandas.read_csv(filepath_or_buffer=data_path / "list_eval_partition.csv",
                    delimiter= ",",
                    header=0,
                    encoding="utf_8")

    train_df = split_df[split_df["partition"] == 0]
    val_df = split_df[split_df["partition"] == 1]
    test_df = split_df[split_df["partition"] == 2]

    split_sets = set(train_df["image_id"]), set(test_df["image_id"]), set(val_df["image_id"])
    set_names = ["train", "test", "val"]

    datasets = {}
    for i, name in enumerate(set_names):
        if name != "test":
            datasets[f"{name}_dataset"] = CelebA(img_path= data_path / "img_align_celeba",
                                                 partition= split_sets[i],
                                                 img_transforms= img_transforms,
                                                 sample_size= sample_size)

        else:
            datasets[f"{name}_dataset"] = CelebA(img_path= data_path / "img_align_celeba",
                                                 partition= split_sets[i],
                                                 img_transforms= img_transforms,
                                                 sample_size = sample_size)

    dataloaders = {}
    for i, name in enumerate(set_names):
        if name != "test":
            dataloaders[f"{name}_dataloader"] = DataLoader(dataset=datasets[f"{name}_dataset"],
                                                           batch_size=batch_size,
                                                           num_workers= num_workers,
                                                           shuffle= True)

        else:
            dataloaders[f"{name}_dataloader"] = DataLoader(dataset=datasets[f"{name}_dataset"],
                                                           batch_size= batch_size,
                                                           num_workers= num_workers,
                                                           shuffle= False)

    return tuple(dataloaders.values()), tuple(datasets.values())
