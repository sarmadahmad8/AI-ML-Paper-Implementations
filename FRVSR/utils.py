import torch
from tqdm.auto import tqdm
from pathlib import Path
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from typing import Dict, List
import matplotlib.pyplot as plt
import random
from kornia.color.ycbcr import RgbToYcbcr, YcbcrToRgb

ycbcr = RgbToYcbcr()
rgb = YcbcrToRgb()

def save_checkpoint(model: torch.nn.Module,
                    checkpoint_name: str,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler = None):

    assert checkpoint_name.endswith(".pth") or checkpoint_name.endswith(".pt"), "Checkpoint name should end with '.pt' or '.pth'"
    
    model_path = Path("models")
    optimizer_path = Path("optimizers")
    scheduler_path = Path("schedulers")

    model_path.mkdir(parents=True, exist_ok=True)
    optimizer_path.mkdir(parents=True, exist_ok=True)
    scheduler_path.mkdir(parents=True, exist_ok=True)

    torch.save(obj = model.state_dict(), f= model_path / checkpoint_name)
    if optimizer:
        torch.save(obj = optimizer.state_dict(), f= optimizer_path / checkpoint_name)
    if scheduler:
        torch.save(obj = scheduler.state_dict(), f= scheduler_path / checkpoint_name)

    print("Model, optimizer, and scheduler are saved")

def load_checkpoint(model: torch.nn.Module,
                    checkpoint_name: str,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                    devce: torch.device = "cuda"):

    assert checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".pth"), "Checkpoint name should end with '.pt' or '.pth'"

    model_path = Path("models")
    optimizer_path = Path("optimizers")
    scheduler_path = Path("schedulers")
    
    model.load_state_dict(torch.load(model_path / checkpoint_name))
    if optimizer:
        optimizer.load_state_dict(torch.load(optimizer_path / checkpoint_name))
    if scheduler:
        scheduler.load_state_dict(torch.load(scheduler_path / checkpoint_name))

def evaluate_model(model: torch.nn.Module,
                   loss_fn: torch.nn.Module,
                   val_dataloader: torch.utils.data.DataLoader,
                   device: torch.device = "cuda",
                   crop_border: int = None):

    val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0

    model.to(device)
    model.eval()
    with torch.inference_mode():
        with tqdm(val_dataloader) as pbar:
            for batch, (X, y) in enumerate(pbar):
                B, T, C, H, W = y.shape
                X, y = X.to(device), y.to(device)
                y = tuple([X, y])
                y_preds = model(X)
                loss = loss_fn(y_preds, y)
                
                y_preds = ycbcr(y_preds[1].view(B * T, C, H, W))
                y = ycbcr(y[1].view(B * T, C, H, W))
                psnr = peak_signal_noise_ratio(y_preds[:, :1, crop_border: -crop_border, crop_border: -crop_border], y[:, :1, crop_border: -crop_border, crop_border: -crop_border])
                ssim = structural_similarity_index_measure(y_preds[:, :1, crop_border: -crop_border, crop_border: -crop_border], y[:, :1, crop_border: -crop_border, crop_border: -crop_border])
                
                val_loss += loss.item()
                val_psnr += psnr.item()
                val_ssim += ssim.item()
    
            val_loss /= len(val_dataloader)
            val_psnr /= len(val_dataloader)
            val_ssim /= len(val_dataloader)

    print(f" Validation metrics:- \nLoss: {val_loss:.5f} | PSNR: {val_psnr:.5f} | SSIM: {val_ssim:.5f}")

def plot_metrics(results: Dict[str, List[float]]):

    epochs = range(len(results["train_loss"]))
    plt.figure(figsize= (12, 30))
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

    rand_idx = random.sample(range(0, len(val_dataset)), k=samples)

    plt.figure(figsize=(60, 20))
    i = 0
    
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for idx in rand_idx:
            X, y = val_dataset[idx]
            T, C, H, W = y.shape
            y = tuple([X, y])
            y_pred = model(X.unsqueeze(dim=0))
    
            y_pred_c = ycbcr(y_pred[1]).view(T, C, H, W)
            y_c = ycbcr(y[1].unsqueeze(dim=0)).view(T, C, H, W)
            psnr = peak_signal_noise_ratio(y_pred_c[:, :1, :, :], y_c[:, :1, :, :])
            ssim = structural_similarity_index_measure(y_pred_c[:, :1, :, :], y_c[:, :1, :, :])
            
            y = y[1].squeeze()
            y = y.clamp(min=0.0, max=1.0)
            y_pred = y_pred[1].squeeze()
            y_pred = y_pred.clamp(min=0.0, max=1.0)

            for l in range(10):
                i += 1
                plt.subplot(samples * 3, 10, i)
                plt.imshow(X[l].permute(1, 2, 0))
                plt.title(f"LR Original, Size: {X[l].shape}")
                plt.axis(False)

            for m in range(10):
                i += 1
                plt.subplot(samples * 3, 10, i)
                plt.imshow(y_pred[m].permute(1, 2, 0))
                plt.title(f"Reconstructed, PSNR: {psnr:.5f} | SSIM: {ssim:.5f}")
                plt.axis(False)

            for n in range(10):
                i += 1
                plt.subplot(samples * 3, 10, i)
                plt.imshow(y[n].permute(1, 2, 0))
                plt.title(f"HR Original, Size: {y[n].shape}")
                plt.axis(False)

    
    #plt.tight_layout(pad=0.1)

    if save_name:
        plt.savefig(fname= image_path / f"{save_name}.jpg",
                    pad_inches= 0.1,
                    dpi= 150)
    plt.show()
