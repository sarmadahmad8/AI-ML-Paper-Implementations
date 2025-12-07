
"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
from pathlib import Path
import random
import matplotlib.pyplot as plt

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    model_target_dir: str,
                    optimizer_target_dir: str,
                    checkpoint_name: str):

    """ Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
                either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model = model_0,
                    target_dir = "models",
                    model_name = "05_going_modular_tinyvgg_model.pth")

    """

    target_dir_path_model = Path(model_target_dir)
    target_dir_path_model.mkdir(parents=True,
                            exist_ok=True)

    target_dir_path_optimizer = Path(optimizer_target_dir)
    target_dir_path_optimizer.mkdir(parents=True,
                            exist_ok=True)


    assert checkpoint_name.endswith(".pth") or checkpoint_name.endswith(".pt"), "checkpoint_name should end with '.pth' or '.pt'."

    model_save_path = target_dir_path_model / checkpoint_name
    optimizer_save_path = target_dir_path_optimizer / checkpoint_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save( obj = model.state_dict(),
                f = model_save_path)
    print(f"[INFO] Saving optimizer to: {optimizer_save_path}")
    torch.save( obj = optimizer.state_dict(),
                f = optimizer_save_path)

def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    checkpoint_name: str,
                   device: torch.device = "cpu"):
    
    assert checkpoint_name.endswith(".pth") or checkpoint_name.endswith(".pt"), "Invalid checkpoint. Checkpoint name should end with ',pth' or '.pt'"
    
    model.load_state_dict(torch.load(f"models/{checkpoint_name}")).to(device)
    optimizer.load_state_dict(torch.load(f"optimzers/{checkpoint_name}"))

    return model, optimizer

def generate_and_display_multiclass(model: torch.nn.Module,
                         dataset: torch.utils.data.dataset,
                         samples: int = 3, 
                         device: torch.device = "cpu"):
    model.to(device)
    model.eval()
    plt.figure(figsize=(samples*4, samples*3))
    rand_idx = random.sample(range(0, len(dataset)), k = samples)
    i=0
    for idx in rand_idx:
        X, y = dataset[idx]
        generated_image = torch.argmax(torch.softmax(model(X.unsqueeze(dim=0).to(device)), dim=1), dim=1)
        plt.subplot(samples, 3, i+1)
        plt.title(f"Original Image")
        plt.imshow(X.permute(1, 2, 0))
        i+=1
        plt.subplot(samples, 3, i+1)
        plt.title(f"Original Mask")
        plt.imshow(y.squeeze())
        i+=1
        plt.subplot(samples, 3, i+1)
        plt.title(f"Generated Mask")
        plt.imshow(generated_image.detach().cpu().squeeze())
        i+=1

def generate_and_display_singleclass(model: torch.nn.Module,
                         dataset: torch.utils.data.dataset,
                         samples: int = 3, 
                         device: torch.device = "cpu"):
    model.to(device)
    model.eval()
    plt.figure(figsize=(samples*4, samples*3))
    rand_idx = random.sample(range(0, len(dataset)), k = samples)
    i=0
    for idx in rand_idx:
        X, y = dataset[idx]
        generated_image = torch.round(torch.sigmoid(model(X.unsqueeze(dim=0).to(device))))
        plt.subplot(samples, 3, i+1)
        plt.title(f"Original Image")
        plt.imshow(X.permute(1, 2, 0))
        i+=1
        plt.subplot(samples, 3, i+1)
        plt.title(f"Original Mask")
        plt.imshow(y.squeeze())
        i+=1
        plt.subplot(samples, 3, i+1)
        plt.title(f"Generated Mask")
        plt.imshow(generated_image.detach().cpu().squeeze())
        i+=1

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"].cpu()
    test_loss = results["test_loss"].cpu()

    accuracy = results["train_acc"].cpu()
    test_accuracy = results["test_acc"].cpu()

    epochs = range(len(results["train_loss"].cpu()))

    plt.figure(figsize=(15, 7))