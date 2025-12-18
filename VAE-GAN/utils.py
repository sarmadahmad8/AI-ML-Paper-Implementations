
import torch
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

def save_checkpoint(checkpoint_name: str,
                    model: torch.nn.Module,
                    discriminator: torch.nn.Module,
                    discriminator_optimizer: torch.optim.Optimizer,
                    decoder_optimizer: torch.optim.Optimizer,
                    encoder_optimizer: torch.optim.Optimizer):

    assert checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".pth"), "Checkpoint name should end with '.pt' or '.pth'."

    model_gan_path = Path("models/gan")
    model_discriminator_path = Path("models/discriminator")
    optimizer_enc_path = Path("optimizers/encoder")
    optimizer_dec_path = Path("optimizers/decoder")
    optimizer_dis_path = Path("optimizers/discriminator")
    
    model_gan_path.mkdir(parents=True,
                         exist_ok=True)
    model_discriminator_path.mkdir(parents=True,
                         exist_ok=True)
    optimizer_enc_path.mkdir(parents=True,
                         exist_ok=True)
    optimizer_dec_path.mkdir(parents=True,
                         exist_ok=True)
    optimizer_dis_path.mkdir(parents=True,
                         exist_ok=True)

    torch.save(obj= model.state_dict(), f= f"models/gan/{checkpoint_name}")
    torch.save(obj= discriminator.state_dict(), f= f"models/discriminator/{checkpoint_name}")
    torch.save(obj= encoder_optimizer.state_dict(), f= f"optimizers/encoder/{checkpoint_name}")
    torch.save(obj= decoder_optimizer.state_dict(), f= f"optimizers/decoder/{checkpoint_name}")
    torch.save(obj= discriminator_optimizer.state_dict(), f= f"optimizers/discriminator/{checkpoint_name}")

def load_checkpoint(checkpoint_name: str,
                   model: torch.nn.Module,
                   discriminator: torch.nn.Module,
                   discriminator_optimizer: torch.optim.Optimizer,
                   decoder_optimizer: torch.optim.Optimizer,
                   encoder_optimizer: torch.optim.Optimizer):
    
    assert checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".pth"), "Checkpoint name should end with '.pt' or '.pth'."
    
    model.load_state_dict(torch.load(f"models/gan/{checkpoint_name}"))
    discriminator.load_state_dict(torch.load(f"models/discriminator/{checkpoint_name}"))
    discriminator_optimizer.load_state_dict(torch.load(f"optimizers/discriminator/{checkpoint_name}"))
    decoder_optimizer.load_state_dict(torch.load(f"optimizers/decoder/{checkpoint_name}"))
    encoder_optimizer.load_state_dict(torch.load(f"optimizers/encoder/{checkpoint_name}"))

def plot_losses(results: Dict[str, List[float]],
                save_name: str = None):

    image_path = Path("images")
    
    epochs = range(0, len(results["encoder_loss"]))
    
    plt.figure(figsize=(16, 16))
    
    plt.subplot(3, 1, 1)
    plt.plot(epochs, results["encoder_loss"], color= "green")
    plt.xlabel("Epochs")
    plt.ylabel("Encoder Loss")
    plt.title("Encoder Loss per Epoch")

    plt.subplot(3, 1, 2)
    plt.plot(epochs, results["decoder_loss"], color= "red")
    plt.xlabel("Epochs")
    plt.ylabel("Decoder Loss")
    plt.title("Decoder Loss per Epoch")

    plt.subplot(3, 1, 3)
    plt.plot(epochs, results["discriminator_loss"], color= "blue")
    plt.xlabel("Epochs")
    plt.ylabel("Discriminator Loss")
    plt.title("Discriminator Loss per Epoch")

    if save_name:
        plt.savefig(fname= image_path / f"{save_name}.jpg",
                    pad_inches=0.1,
                    dpi=60)

    plt.show()

def plot_image_grid(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    grid_dim: int = 8,
                    device: torch.device = "cuda",
                   save_name: str = None):
    
    image_path = Path("images")
    image_path.mkdir(parents=True,
                     exist_ok=True)

    model.to(device)
    model.eval()
    full_batch = []
    for batch, (X, _) in enumerate(dataloader):
        with torch.inference_mode():
            if grid_dim ** 2 >= batch * X.shape[0]:
                X = X.to(device)
                _, _, _, reconstruction_batch = model(X)
                full_batch.append(reconstruction_batch)

            else:
                break
                
    full_batch_tensor = torch.cat([batch for batch in full_batch], dim=0)
    
    plt.figure(figsize=(64, 64))
    for i in range(grid_dim * grid_dim):
        plt.subplot(grid_dim, grid_dim, i+1)
        img= full_batch_tensor[i].permute(1, 2, 0).detach().cpu()
        img = ((img + 1) / 2)
        plt.imshow(img)
        plt.axis(False)

    plt.tight_layout(pad=0.1)
    if save_name:
        plt.savefig(fname= image_path / f"{save_name}.jpg",
                   dpi = 30,
                   pad_inches=0.1)
        
    plt.show()
    plt.close()
    
