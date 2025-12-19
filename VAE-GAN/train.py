import torch
from torchvision.transforms import v2
from utils import save_checkpoint, plot_losses, plot_image_grid
from model import VAEGAN
from engine import train
from download_data import download_data
from data_setup import create_dataloaders

data_path = download_data(dataset_name= "Celeba")

img_transforms = v2.Compose([
    v2.Resize(size=(64, 64),
              interpolation=v2.InterpolationMode.BILINEAR),
    v2.ToImage(),
    v2.ToDtype(dtype= torch.float32, 
               scale=True),
    v2.Normalize(mean= [0.5, 0.5, 0.5], 
                 std= [0.5, 0.5, 0.5])
])

(train_dataloader, test_dataloader, val_dataloader), (train_dataset, test_dataset, val_dataset) = create_dataloaders(data_path= data_path,
                                                                                                                 img_transforms= img_transforms,
                                                                                                                sample_size = 0.5,
                                                                                                                 batch_size= 64,
                                                                                                                 num_workers= 4)

print(len(train_dataloader), len(test_dataloader), len(val_dataloader), len(train_dataset), len(test_dataset), len(val_dataset))

device = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
EPOCHS = 25
GAMMA = 20
ALPHA = 0.1

model = VAEGAN(in_channels=3,
               latent_dim=128,
               out_channels=3).to(device= device)

train_results, test_results = train(model = model,
                                    train_dataloader= train_dataloader,
                                    test_dataloader= test_dataloader,
                                    lr = LR,
                                    epochs = EPOCHS,
                                    gamma = GAMMA,
                                    alpha= ALPHA,
                                    device = device)

plot_losses(results= train_results)
plot_losses(results= test_results)
