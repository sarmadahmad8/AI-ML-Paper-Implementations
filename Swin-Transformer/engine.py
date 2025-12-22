
"""
Containsfunctions for traning and testing PyTorch model.
"""

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all of 
    the required training steps.

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss.
    device: A target device to compute on (e.g "cuda" or "cpu")

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form of (training_loss, training_acc). FOr example:

    (0.1112, 0.8743)
    """

    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        y_preds_labels = torch.argmax(torch.softmax(y_preds, dim=1), dim=1)
        loss = loss_fn(y_preds, y)
        train_loss += loss.item()
        acc = torch.sum(y_preds_labels==y).item()/len(y_preds_labels)
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to evaluation mode and then runs through all of 
    the required testing steps.

    Args:
    model: A PyTorch model to be evaluated.
    dataloader: A DataLoader instance for the model to be evaluated on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss.
    device: A target device to compute on (e.g "cuda" or "cpu")

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form of (training_loss, training_acc). FOr example:

    (0.1112, 0.8743)
    """
    
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(dataloader)):
            X, y = X.to(device), y.to(device)
            test_preds = model(X)
            test_preds_labels = torch.argmax(torch.softmax(test_preds, dim=1),dim=1)
            loss = loss_fn(test_preds, y)
            acc = torch.sum(test_preds_labels==y).item()/len(test_preds_labels)
            test_loss += loss.item()
            test_acc += acc

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:

    """Trains a PyTorch model for the number of input epochs.

    Passes a target PyTorch model through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Args:
    model: A PyTorch model to be trained.
    train_dataloader: A PyTOrch dataloader for the model to be trained on.
    test_dataloader: A PyTorch dataloader for the model to be evaluated on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g "cuda" or "cpu")

    Returns:
    A dictionary containing the training and testing loss as well as accuracy
    metrics. Each metric has a value in a list for each epoch.
    In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}

    For example if training for epochs = 3:
                {train_loss: [2.0619, 1.0537, 0.9124],
                train_acc: [0.39, 0.42, 0.56],
                test_loss: [2.1319, 1.5667, 1.1234],
                test_acc: [0.37, 0.48, 0.67]}
    """
    results = {"train_loss" :[],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model = model,
                                           dataloader= train_dataloader,
                                           loss_fn= loss_fn,
                                           optimizer= optimizer,
                                           device = device)
        test_loss, test_acc = test_step(model = model,
                                        dataloader = test_dataloader,
                                        loss_fn = loss_fn,
                                        device = device)

        print(f" Epoch {epoch + 1} |"
                f" train_loss: {train_loss:.4f} |"
                f" train_acc: {train_acc:.2f} |"
                f" test_loss: {test_loss:.4f} |"
                f" test_acc: {test_acc:.2f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
