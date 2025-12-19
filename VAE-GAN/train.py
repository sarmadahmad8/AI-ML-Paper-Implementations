import torch
from torchvision.transforms import v2
from utils import save_checkpoint, plot_losses, plot_image_grid
from model import VAEGAN
from engine import train
from download_data import download_data
from data_setup import create_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
EPOCHS = 25
GAMMA = 20
ALPHA = 0.1
SAMPLE_SIZE = 0.5
BATCH_SIZE = 64
checkpoint_name = f"VAE-GAN-{EPOCHS}epoch-{SAMPLE_SIZE}sample_{LR}lr.pth"

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
                                                                                                                sample_size = SAMPLE_SIZE,
                                                                                                                 batch_size= BATCH_SIZE,
                                                                                                                 num_workers= 4)

print(len(train_dataloader), len(test_dataloader), len(val_dataloader), len(train_dataset), len(test_dataset), len(val_dataset))


model = VAEGAN(in_channels=3,
               latent_dim=128,
               out_channels=3).to(device= device)

discriminator_optimizer = torch.optim.RMSprop(model.discriminator.parameters(), 
                                              lr = LR * ALPHA)
encoder_optimizer = torch.optim.RMSprop(model.encoder.parameters(),
                                        lr = LR)
decoder_optimizer = torch.optim.RMSprop(model.decoder.parameters(),
                                        lr = LR)

train_results, test_results = train(model = model,
                                    train_dataloader= train_dataloader,
                                    test_dataloader= test_dataloader,
                                    discriminator_optimizer= discriminator_optimizer,
                                    decoder_optimizer= decoder_optimizer,
                                    encoder_optimizer= encoder_optimizer,
                                    epochs = EPOCHS,
                                    gamma = GAMMA,
                                    device = device)

plot_losses(results= train_results)
plot_losses(results= test_results)

save_checkpoint(model= model,
                discriminator= model.discriminator,
                discriminator_optimizer= discriminator_optimizer,
                decoder_optimizer= decoder_optimizer,
                encoder_optimizer= encoder_optimizer,
                checkpoint_name= checkpoint_name)
