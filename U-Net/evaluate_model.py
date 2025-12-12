"""
Contains functions to evaluate both binary class and multiclass datasets.
"""

import torch
import torch.nn as nn
from typing import Dict, List
import argparse
from model import UNet

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type= str, help="pass the trained model file path to evaluate (e.g 'models/UNet-30epochs.pth')")
    parser.add_argument("--dataloader", type= torch.utils.data.DataLoader, help="pass the validation dataloader to evaluate on")
    parser.add_argument("--device", type=torch.device, help="pass the device to compute on (e.g 'cpu' or 'cuda')")
    args = parser.parse_args()
    model_0 = UNet(in_channels=3,
                   out_channels=34)
    model_0.load_state_dict(torch.load(args.model))
    results = evaluate_model(model=model_0,
                             dataloader=args.dataloader,
                             device = args.device)
