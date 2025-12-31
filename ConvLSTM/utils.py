import torch
from pathlib import Path
import matplotlib.pyplot as plt
import random
from typing import Dict, List

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

def plot_reconstructed_sequence(model: torch.nn.Module,
                                dataset: torch.utils.data.Dataset,
                                samples: int = 2,
                                device: torch.device = "cpu"):

    model.to(device)
    model.eval()
    rand_idx = random.sample(range(0, len(dataset)), k= samples)
    plt.figure(figsize=(samples * 5, samples * 5))
    k = 1
    
    for i, idx in enumerate(rand_idx):
        with torch.inference_mode():
            X = dataset[idx]
            X, y = X[:5].unsqueeze(dim = 0), X[5:].unsqueeze(dim = 0)
            X, y = X.to(device), y.to(device)
            y_preds = model(X)
            y_preds_labels = torch.round(torch.sigmoid(y_preds)).float()
            X, y, y_preds_labels = X.squeeze(dim = 0), y.squeeze(dim = 0), y_preds_labels.squeeze(dim = 0)
            for j in range(5):
                plt.subplot(3 * samples, 5, k)
                plt.imshow(X[j].permute(1, 2, 0))
                plt.axis(False)
                plt.title(f"X_t-1")
                k += 1
            plt.show()
            
            for j in range(5):
                plt.subplot(3 * samples, 5, k)
                plt.imshow(y[j].permute(1, 2, 0))
                plt.axis(False)
                plt.title(f"X_t+1")
                k += 1
            plt.show()

            for j in range(5):
                plt.subplot(3 * samples, 5, k)
                plt.imshow(y_preds_labels[j].permute(1, 2, 0))
                plt.axis(False)
                plt.title(f"X_tilda_t+1")
                k += 1
    
    plt.show()
            
