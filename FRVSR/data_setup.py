import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from PIL import Image
from typing import Tuple
from pathlib import Path

class CustomVSRDataset(Dataset):

    def __init__(self,
                 img_dir: str,
                 sample_size: float = 0.1,
                 patch_size: int = 256,
                 seq_length: int = 10,
                 stride: int = 10,
                 img_extension: str = "jpg"):
    
        self.patch_size = patch_size
        self.seq_length = seq_length
        self.stride = stride
        
        self.img_path_list = sorted(list(img_dir.glob(f"*.{img_extension}")))
        self.sample_size = int(len(self.img_path_list) * sample_size)
        self.img_path_list = self.img_path_list[: self.sample_size]
        
        self.width, self.height = Image.open(self.img_path_list[0]).convert(mode="RGB").size
        self.vertical_patches = self.height // self.patch_size
        self.horizontal_patches = self.width // self.patch_size
        self.patches_per_image = self.vertical_patches * self.horizontal_patches
        self.num_sequences = (len(self.img_path_list) - self.seq_length) // self.stride + 1
        self.total_sample_sequences = self.num_sequences * self.patches_per_image

    def get_patch_coordinates(self,
                   patch_index: int) -> Tuple[int, int, int, int]:

        row = patch_index // self.horizontal_patches
        col = patch_index % self.horizontal_patches

        y_start = row * self.patch_size
        y_end = y_start + self.patch_size
        x_start = col * self.patch_size
        x_end = x_start + self.patch_size

        return y_start, y_end, x_start, x_end

    def _get_lr_sequence(self,
                         sequence: torch.Tensor) -> torch.Tensor:

        gaussian_blur = v2.GaussianBlur(kernel_size=(5, 5),
                                        sigma=(1.5, 1.5))
        
        hr_image_noise = gaussian_blur(sequence)
        lr_sequence = hr_image_noise[:, :, ::4, ::4]

        return lr_sequence

    def _transforms(self,
                    lr_sequence: torch.Tensor,
                    hr_sequence: torch.Tensor):

        transforms = v2.Compose([
            v2.ToDtype(dtype=torch.float32,
                       scale=True)
        ])

        return transforms(lr_sequence), transforms(hr_sequence)

    def __len__(self):

        return self.total_sample_sequences

    def __getitem__(self,
                    index: int):

        sequence_index = index // self.patches_per_image
        patch_index = index % self.patches_per_image

        start_frame = sequence_index * self.stride

        y_start, y_end, x_start, x_end = self.get_patch_coordinates(patch_index= patch_index)

        sequence = []
        for t in range(self.seq_length):

            frame_idx = start_frame + t
            img = Image.open(self.img_path_list[frame_idx]).convert(mode="RGB")
            frame = v2.PILToTensor()(img)
            # print(frame.shape, frame.min(), frame.max(), frame.dtype)
            patch = frame[:, y_start: y_end, x_start: x_end]
            # print(patch.shape, patch.min(), patch.max(), patch.dtype)
            sequence.append(patch)

        hr_sequence = torch.stack(sequence, dim=0)
        # print(hr_sequence.min(), hr_sequence.max())
        lr_sequence = self._get_lr_sequence(sequence= hr_sequence)

        lr_sequence, hr_sequence = self._transforms(lr_sequence=lr_sequence,
                                                    hr_sequence=hr_sequence)

        return lr_sequence, hr_sequence

def create_dataloaders_custom(img_dir: str,
                              sample_size: float = 0.1,
                              patch_size: int = 256,
                              seq_length: int = 10,
                              stride: int = 10,
                              img_extension: str = "jpg",
                              batch_size: int = 1,
                              num_workers: int = 8):

    dataset = CustomVSRDataset(img_dir=img_dir,
                               sample_size= sample_size,
                               patch_size= patch_size,
                               seq_length=seq_length,
                               stride=stride,
                               img_extension=img_extension)

    train_length = int(len(dataset) * 0.8)
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset= dataset,
                                               lengths=[train_length, test_length])

    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size=batch_size,
                                  num_workers= num_workers,
                                  shuffle= True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)

    return train_dataloader, test_dataloader, train_dataset, test_dataset

class Vid4Dataset(Dataset):

    def __init__(self,
                 img_dir: str,
                 seq_length: int,
                 stride: int,
                 patch_size: int,
                 img_extension: str = "png"):

        self.seq_length = seq_length
        self.stride = stride
        self.patch_size = patch_size

        self.video_frames = []
        self.video_info = []
        self.patches_per_frame = []
        self.total_sample_sequences = 0
        
        for dir_ in img_dir.iterdir():
            if dir_.is_dir():
                frames = sorted(list(dir_.glob(f"*.{img_extension}")))

            self.video_frames.append(frames)

        for i, _ in enumerate(self.video_frames):
            first_img = Image.open(self.video_frames[i][0]).convert(mode="RGB")
            self.width, self.height = first_img.size
            self.vertical_patches = self.height // self.patch_size
            self.horizontal_patches = self.width // self.patch_size
            self.patches_per_frame.append(self.vertical_patches * self.horizontal_patches)

        for i, frames in enumerate(self.video_frames):
            num_sequences = (len(frames) - self.seq_length) // self.stride + 1
            video_samples = num_sequences * self.patches_per_frame[i]

            self.video_info.append({
                "start_idx": self.total_sample_sequences,
                "num_sequences": num_sequences,
                "num_frames": len(frames),
                "samples": video_samples})
                                   

            self.total_sample_sequences += video_samples

    def get_patch_coordinates(self,
                               patch_index: int) -> Tuple[int, int, int, int]:

        row = patch_index // self.horizontal_patches
        col = patch_index % self.horizontal_patches

        y_start = row * self.patch_size
        y_end = y_start + self.patch_size
        x_start = col * self.patch_size
        x_end = x_start + self.patch_size

        return y_start, y_end, x_start, x_end

    def _get_lr_sequence(self,
                         sequence: torch.Tensor) -> torch.Tensor:

        gaussian_blur = v2.GaussianBlur(kernel_size=(5, 5),
                                        sigma=(1.5, 1.5))
        
        hr_image_noise = gaussian_blur(sequence)
        lr_sequence = hr_image_noise[:, :, ::4, ::4]

        return lr_sequence

    def _transforms(self,
                    lr_sequence: torch.Tensor,
                    hr_sequence: torch.Tensor):

        transforms = v2.Compose([
            v2.ToDtype(dtype=torch.float32,
                       scale=True)
        ])

        return transforms(lr_sequence), transforms(hr_sequence)

    def __len__(self):

        return self.total_sample_sequences

    def __getitem__(self,
                    index: int):

        video_idx = 0
        local_index = index

        for i, info in enumerate(self.video_info):
            if index < info["start_idx"] + info["samples"]:
                video_idx = i
                local_index = index - info["start_idx"]
                break

        frames = self.video_frames[video_idx]

        sequence_index = local_index // self.patches_per_frame[video_idx]
        patch_index = local_index % self.patches_per_frame[video_idx]

        start_frame = sequence_index * self.stride

        y_start, y_end, x_start, x_end = self.get_patch_coordinates(patch_index= patch_index)

        sequence = []
        for t in range(self.seq_length):

            frame_idx = start_frame + t
            img = Image.open(frames[frame_idx]).convert(mode="RGB")
            frame = v2.PILToTensor()(img)
            # print(frame.shape, frame.min(), frame.max(), frame.dtype)
            patch = frame[:, y_start: y_end, x_start: x_end]
            # print(patch.shape, patch.min(), patch.max(), patch.dtype)
            sequence.append(patch)

        hr_sequence = torch.stack(sequence, dim=0)
        # print(hr_sequence.min(), hr_sequence.max())
        lr_sequence = self._get_lr_sequence(sequence= hr_sequence)

        lr_sequence, hr_sequence = self._transforms(lr_sequence=lr_sequence,
                                                    hr_sequence=hr_sequence)

        return lr_sequence, hr_sequence

def create_dataloaders_vid4(img_dir: str,
                            patch_size: int = 256,
                            seq_length: int = 10,
                            stride: int = 10,
                            img_extension: str = "png",
                            batch_size: int = 1,
                            num_workers: int = 8):

    val_dataset = Vid4Dataset(img_dir=img_dir,
                              patch_size= patch_size,
                              seq_length=seq_length,
                              stride=stride,
                              img_extension=img_extension)

    val_dataloader = DataLoader(dataset= val_dataset,
                                batch_size=batch_size,
                                num_workers= num_workers,
                                shuffle= True)

    return val_dataloader, val_dataset
