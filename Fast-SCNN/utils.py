
"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
import torch.nn as nn
from pathlib import Path
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

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

    """Loads a saved checkpoint to a PyTorch model.

    Args:
    model: A PyTorch model to load.
    optimizer: A PyTorch optimizer to load.
    checkpoint_name: The name of the checkpoint you want to load.
    device: The device to move the model to (e.g "cuda", "cpu")

    Returns:
    model: A PyTorch model with loaded state.
    optimizer: A PyTorch optimizer with loaded state.

    """
    
    assert checkpoint_name.endswith(".pth") or checkpoint_name.endswith(".pt"), "Invalid checkpoint. Checkpoint name should end with '.pth' or '.pt'"
    
    model.load_state_dict(torch.load(f"models/{checkpoint_name}"))
    model.to(device)
    optimizer.load_state_dict(torch.load(f"optimizers/{checkpoint_name}"))

    return model, optimizer

def generate_and_display_multiclass(model: torch.nn.Module,
                         dataset: torch.utils.data.dataset,
                         samples: int = 3, 
                         device: torch.device = "cpu"):

    """
    Generates and displays samples using a trained model checkpoint.

    Args:
    model: A PyTorch model.
    dataset: A PyTorch dataset to sample from.
    samples: An integer containing the number of samples to display
    device: The device to perform computation on (e.g "cuda", "cpu")
    """
    
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
        plt.axis(False)
        i+=1
        plt.subplot(samples, 3, i+1)
        plt.title(f"Original Mask")
        plt.imshow(y.squeeze())
        plt.axis(False)
        i+=1
        plt.subplot(samples, 3, i+1)
        plt.title(f"Generated Mask")
        plt.imshow(generated_image.detach().cpu().squeeze().long())
        plt.axis(False)
        i+=1

def generate_and_display_singleclass(model: torch.nn.Module,
                         dataset: torch.utils.data.dataset,
                         samples: int = 3, 
                         device: torch.device = "cpu"):

    """
    Generates and displays samples using a trained model checkpoint.

    Args:
    model: A PyTorch model.
    dataset: A PyTorch dataset to sample from.
    samples: An integer containing the number of samples to display
    device: The device to perform computation on (e.g "cuda", "cpu")
    """
    
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
        plt.axis(False)
        i+=1
        plt.subplot(samples, 3, i+1)
        plt.title(f"Original Mask")
        plt.imshow(y.squeeze())
        plt.axis(False)
        i+=1
        plt.subplot(samples, 3, i+1)
        plt.title(f"Generated Mask")
        plt.imshow(generated_image.detach().cpu().squeeze())
        plt.axis(False)
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

def evaluate_model_multiclass(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
                   device: torch.device = "cpu") -> Dict[str, List[float]]:

    """
    Evalutes a PyTorch models performance on a validation dataloader.

    Args:
    model: A PyTorch model to evaluate.
    dataloader: A validation dataloader to evaluate the model on.
    loss_fn: A PyTorch loss function to use.
    device: The device to perform computation on (e.g "cuda", "cpu")

    Returns:
    results: A dictionary containing the validation loss and validation accuracy.

    Example usage:
    evaluate_model_class(model = model_0,
                        dataloader = val_dataloader,
                        loss_fn = nn.CrossEntropyLoss(),
                        device = device)
    """

    val_loss, val_acc = 0, 0
    results = {"val_loss":[],
               "val_acc": []
              }
    
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            val_preds = model(X)
            val_preds_labels = torch.argmax(torch.softmax(val_preds, dim= 1), dim= 1)
            # print(val_preds.shape, y.shape)
            loss = loss_fn(val_preds, y.squeeze(dim=1))
            acc = torch.sum(val_preds_labels==y.squeeze(dim=1))/val_preds_labels.numel()
            val_loss += loss.item()
            val_acc += acc.item()

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)
    print(f"Evaluation Loss: {val_loss:.5f} | Evaluation Accuracy: {val_acc:.2f}%")

    return results

def evaluate_model_binaryclass(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
                   device: torch.device = "cpu",
                    weight_map: bool = False) -> Dict[str, List[float]]:

    """
    Evalutes a PyTorch models performance on a validation dataloader.

    Args:
    model: A PyTorch model to evaluate.
    dataloader: A validation dataloader to evaluate the model on.
    loss_fn: A PyTorch loss function to use.
    device: The device to perform computation on (e.g "cuda", "cpu")

    Returns:
    results: A dictionary containing the validation loss and validation accuracy.

    Example usage:
    evaluate_model_class(model = model_0,
                        dataloader = val_dataloader,
                        loss_fn = nn.BCEWithLogitsLoss(),
                        device = device)
    """

    val_loss, val_acc = 0, 0
    results = {"val_loss":[],
               "val_acc": []
              }

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            val_preds = model(X)
            val_preds_labels = torch.round(torch.sigmoid(val_preds)).type(torch.int32)
            if weight_map:
                loss = loss_fn(val_preds.squeeze(dim=1), (y*255).squeeze(dim=1)).mean()
            else:
                loss = loss_fn(val_preds.squeeze(dim=1), (y*255).squeeze(dim=1))
            acc = torch.sum(val_preds_labels.squeeze()==(y*255).squeeze(dim=1).type(torch.int32))/val_preds_labels.numel()
            val_loss += loss.item()
            val_acc += acc.item()

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)

    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)
    print(f"Evaluation Loss: {val_loss:.5f} | Evaluation Accuracy: {val_acc:.2f}%")

    return results

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def tune_parameters(model: torch.nn.Module):
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name:
            param.requires_grad=False

        if "pointwise" in name:
            decay_params.append(param)

        else:
            no_decay_params.append(param)

    return decay_weights, no_decay_weights