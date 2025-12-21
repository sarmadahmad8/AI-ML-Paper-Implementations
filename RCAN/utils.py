import torch
from tqdm.auto import tqdm
from pathlib import Path
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from typing import Dict, List
import matplotlib.pyplot as plt
import random

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    checkpoint_name: str):

    assert checkpoint_name.endswith(".pth") or checkpoint_name.endswith(".pt"), "Checkpoint name should end with '.pt' or '.pth'"
    
    model_path = Path("models")
    optimizer_path = Path("optimizers")
    scheduler_path = Path("schedulers")

    model_path.mkdir(parents=True, exist_ok=True)
    optimizer_path.mkdir(parents=True, exist_ok=True)
    scheduler_path.mkdir(parents=True, exist_ok=True)

    torch.save(obj = model.state_dict(), f= model_path / checkpoint_name)
    torch.save(obj = optimizer.state_dict(), f= optimizer_path / checkpoint_name)
    torch.save(obj = scheduler.state_dict(), f= scheduler_path / checkpoint_name)

    print("Model, optimizer, and scheduler are saved")

def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    checkpoint_name: str,
                    devce: torch.device = "cuda"):

    assert checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".pth"), "Checkpoint name should end with '.pt' or '.pth'"

    model_path = Path("models")
    optimizer_path = Path("optimizers")
    scheduler_path = Path("schedulers")
    
    model.load_state_dict(torch.load(model_path / checkpoint_name))
    optimizer.load_state_dict(torch.load(optimizer_path / checkpoint_name))
    scheduler.load_state_dict(torch.load(scheduler_path / checkpoint_name))

def evaluate_model(model: torch.nn.Module,
                   loss_fn: torch.nn.Module,
                   val_dataloader: torch.utils.data.DataLoader,
                   device: torch.device = "cuda"):

    val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(val_dataloader)):
            X, y = X.to(device), y.to(device)
            y_preds = model(X)
            loss = loss_fn(y_preds, y)
            psnr = peak_signal_noise_ratio(y_preds, y)
            ssim = structural_similarity_index_measure(y_preds, y)

            val_loss += loss
            val_psnr += psnr
            val_ssim += ssim

        val_loss /= len(val_dataloader)
        val_psnr /= len(val_dataloader)
        val_ssim /= len(val_dataloader)

    print(f" Validation metrics:- \nLoss: {val_loss:.5f} | PSNR: {val_psnr:.5f} | SSIM: {val_ssim:.5f}")

def plot_metrics(results: Dict[str, List[float]]):

    epochs = range(len(results))
    plt.figure(figsize= (20, 16))
    for i, (name, metric) in enumerate(results.items()):
        plt.subplot(len(results), 1, i+1)
        plt.plot(epochs, metric)
        plt.xlabel("epochs")
        plt.ylabel(name.split("_")[1])
        plt.title(f"{name.split('_')[0]} {name.split('_')[1]} per epoch")

    plt.show()

def plot_reconstructed_images(model: torch.nn.Module,
                              val_dataset: torch.utils.data.Dataset,
                              samples: int = 5,
                              device: torch.device = "cpu",
                              save_name: str = None):
    
    image_path = Path("images")
    image_path.mkdir(parents=True,
                     exist_ok=True)

    rand_idx = random.sample(range(0, len(val_dataset)), k=5)

    plt.figure(figsize=(8, 20))
    i = 0
    
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for idx in rand_idx:
            X, y = val_dataset[idx]
            y_pred = model(X.unsqueeze(dim=0))
    
            psnr = peak_signal_noise_ratio(y_pred, y.unsqueeze(dim=0))
            ssim = structural_similarity_index_measure(y_pred, y.unsqueeze(dim=0))

            y = y.squeeze()
            y = y.clamp(min=0.0, max=1.0)
            y_pred = y_pred.squeeze()
            y_pred = y_pred.clamp(min=0.0, max=1.0)
            
            i += 1
            plt.subplot(samples, 2, i)
            plt.imshow(y.permute(1, 2, 0))
            plt.title("Original")
            plt.axis(False)
    
            i += 1
            plt.subplot(samples, 2, i)
            plt.imshow(y_pred.permute(1, 2, 0))
            plt.title(f"PSNR: {psnr:.5f} | SSIM: {ssim:.5f}")
            plt.axis(False)

    
    plt.tight_layout(pad=0.1)

    if save_name:
        plt.savefig(fname= image_path / f"{save_name}.jpg",
                    pad_inches= 0.1,
                    dpi= 150)
    plt.show()
