"""
Contains functions for traning and testing PyTorch model.
"""

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from skimage.measure import label

class WeightMap:
    def __init__(self,
                 w0=10,
                 sigma=5):
        self.w0 = w0
        self.sigma = sigma

    def compute_wc(self, mask):
        # binary mask → class weights
        classes, counts = np.unique(mask, return_counts=True)
        freq = counts / counts.sum()
        class_weight = {cls: 1.0 / freq[i] for i, cls in enumerate(classes)}

        wc = np.zeros_like(mask, dtype=np.float32)
        for cls, w in class_weight.items():
            wc[mask == cls] = w
        return wc

    def compute_d1_d2(self,
                      instance_mask):
        h, w = instance_mask.shape
        borders = []

        instance_ids = [i for i in np.unique(instance_mask) if i != 0]

        for inst in instance_ids:
            obj = (instance_mask == inst)
            eroded = binary_erosion(obj)
            border = obj ^ eroded
            borders.append(border)

        if len(borders) == 0:
            return np.zeros((h, w)), np.zeros((h, w))

        distances = np.stack([distance_transform_edt(~b) for b in borders], axis=0)
        distance_sorted = np.sort(distances, axis=0)

        d1 = distance_sorted[0]
        d2 = distance_sorted[1] if len(distance_sorted) > 1 else np.full_like(d1, 1e-6)

        return d1, d2

    def compute_weight_map(self,
                           binary_mask):
        instance_mask = label(binary_mask)

        wc = self.compute_wc(binary_mask)
        d1, d2 = self.compute_d1_d2(instance_mask)

        boundary_term = self.w0 * np.exp(-((d1 + d2)**2) / (2 * (self.sigma**2)))
        return (wc + boundary_term).astype(np.float32)

def train_step_ISBI(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               weight_map: bool = False) -> Tuple[float, float]:
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
    wm = WeightMap(w0=10, sigma=5)
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        y_preds_labels = torch.sigmoid(y_preds)
        binary_preds_labels = torch.round(y_preds_labels).type(torch.int32)
        if weight_map:
            wmap = wm.compute_weight_map((y*255).squeeze().cpu().numpy())
            loss = (torch.tensor(wmap).unsqueeze(dim=0).to(device) * loss_fn(y_preds.squeeze(dim=1), (y*255).squeeze(dim=1))).mean()
        else:
            loss = loss_fn(y_preds.squeeze(), (y*255).squeeze())
            
        train_loss += loss.item()
        acc = torch.sum(binary_preds_labels==(y*255).type(torch.int32)).item()/y.numel()
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step_ISBI(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
               weight_map: bool = False) -> Tuple[float, float]:

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
    wm = WeightMap(w0=10, sigma=5)
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_preds = model(X)
            test_preds_labels = torch.sigmoid(test_preds)
            if weight_map:
                wmap = wm.compute_weight_map((y*255).squeeze().cpu().numpy())
                loss = (torch.tensor(wmap).unsqueeze(dim=0).to(device) * loss_fn(test_preds.squeeze(dim=1), (y*255).squeeze(dim=1))).mean()
            else:
                loss = loss_fn(test_preds.squeeze(), (y*255).squeeze())
            test_preds_labels = torch.round(test_preds_labels).type(torch.int32)
            acc = torch.sum(test_preds_labels==(y*255).type(torch.int32)).item()/test_preds_labels.numel()
            test_loss += loss.item()
            test_acc += acc

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

def train_ISBI(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          weight_map: bool = False) -> Dict[str, List[float]]:

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
        train_loss, train_acc = train_step_ISBI(model = model,
                                           dataloader= train_dataloader,
                                           loss_fn= loss_fn,
                                           optimizer= optimizer,
                                           device = device,
                                            weight_map = weight_map)
        test_loss, test_acc = test_step_ISBI(model = model,
                                        dataloader = test_dataloader,
                                        loss_fn = loss_fn,
                                        device = device,
                                        weight_map = weight_map)

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

def train_step_CS(model: torch.nn.Module,
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
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        y_preds_labels = torch.argmax(torch.softmax(y_preds, dim=1), dim=1)
        loss = loss_fn(y_preds, y.squeeze(dim=1))
        train_loss += loss.item()
        # print(y_preds_labels.unique())
        acc = torch.sum(y_preds_labels == y.squeeze(dim=1))/y_preds_labels.numel()
        train_acc += acc.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(y_preds_labels.unique())
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step_CS(model: torch.nn.Module,
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
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_preds = model(X)
            test_preds_labels = torch.argmax(torch.softmax(test_preds, dim=1), dim=1)
            loss = loss_fn(test_preds, y.squeeze(dim=1))
            acc = torch.sum(test_preds_labels == y.squeeze(dim=1))/test_preds_labels.numel()
            test_loss += loss.item()
            test_acc += acc.item()

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

def train_CS(model: torch.nn.Module,
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
        train_loss, train_acc = train_step_CS(model = model,
                                           dataloader= train_dataloader,
                                           loss_fn= loss_fn,
                                           optimizer= optimizer,
                                           device = device)
        test_loss, test_acc = test_step_CS(model = model,
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

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step_CV(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               weight_map: bool = False) -> Tuple[float, float]:
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
    wm = WeightMap(w0=10, sigma=5)
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        y_preds_labels = torch.sigmoid(y_preds)
        binary_preds_labels = torch.round(y_preds_labels).type(torch.int32)
        if weight_map:
            wmap = wm.compute_weight_map((y*255).squeeze().cpu().numpy())
            loss = (torch.tensor(wmap).unsqueeze(dim=0).to(device) * loss_fn(y_preds.squeeze(dim=1), (y*255).squeeze(dim=1))).mean()
        else:
            loss = loss_fn(y_preds.squeeze(dim=1), (y*255).squeeze(dim=1))
        print(y_preds_labels.type(torch.int32).unique())
        train_loss += loss
        acc = torch.sum(binary_preds_labels.squeeze()==(y*255).squeeze().type(torch.int32)).item()/y.numel()
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step_CV(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              weight_map: bool = False) -> Tuple[float, float]:

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
    wm = WeightMap(w0=10, sigma=5)
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_preds = model(X)
            test_preds_labels = torch.sigmoid(test_preds)
            test_preds_labels = torch.round(test_preds_labels).type(torch.int32)
            if weight_map:
                wmap = wm.compute_weight_map((y*255).squeeze().cpu().numpy())
                loss = (torch.tensor(wmap).unsqueeze(dim=0).to(device) * loss_fn(test_preds.squeeze(dim=1), (y*255).squeeze(dim=1))).mean()
            else:
                loss = loss_fn(test_preds.squeeze(dim=1), (y*255).squeeze(dim=1))
            test_preds_labels = torch.round(test_preds_labels).type(torch.int32)
            acc = torch.sum(test_preds_labels.squeeze()==(y*255).squeeze().type(torch.int32)).item()/y.numel()
            test_loss += loss
            test_acc += acc

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

def train_CV(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
            weight_map: bool = False) -> Dict[str, List[float]]:

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
        train_loss, train_acc = train_step_CV(model = model,
                                           dataloader= train_dataloader,
                                           loss_fn= loss_fn,
                                           optimizer= optimizer,
                                           device = device,
                                             weight_map = weight_map)
        test_loss, test_acc = test_step_CV(model = model,
                                        dataloader = test_dataloader,
                                        loss_fn = loss_fn,
                                        device = device,
                                          weight_map = weight_map)

        print(f" Epoch {epoch + 1} |"
                f" train_loss: {train_loss:.4f} |"
                f" train_acc: {train_acc:.4f} |"
                f" test_loss: {test_loss:.4f} |"
                f" test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
