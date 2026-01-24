import torch
from torch import autocast, GradScaler
from torchvision.transforms import v2
import torch.nn.functional as F
from typing import Tuple
from tqdm.auto import tqdm
from utils import save_checkpoint

def scale_flow(flow: torch.Tensor,
               original_height: int,
               original_width: int,
               new_height: int,
               new_width: int) -> torch.Tensor:

    scale_h = new_height / original_height
    scale_w = new_width / original_width
    
    flow_resized = F.interpolate(flow, 
                                  size=(new_height, new_width),
                                  mode='bilinear',
                                  align_corners=True)
    
    flow_resized[:, 0, :, :] *= scale_w
    flow_resized[:, 1, :, :] *= scale_h
    
    return flow_resized

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               resize: Tuple[int, int],
               scheduler: torch.optim.lr_scheduler.LRScheduler = None,
               scaler: torch.amp.GradScaler = None,
               device: torch.device = "cuda"):

    resized = v2.Resize(size=resize)
    normalize = v2.Normalize(mean= [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     std= [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    model.train()
    train_l1_loss, train_epe, train_1px, train_3px, train_5px = 0.0, 0.0, 0.0, 0.0, 0.0

    for batch, X in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        
        img_1, img_2, flow = X
        X, y = torch.cat((img_1, img_2), dim= 1), flow.permute(0, 2, 3, 1)
        X, y = X.to(device), y.to(device)
        X, y = normalize(resized(X)), scale_flow(flow=y,
                                                 original_height = y.shape[-2],
                                                 original_width = y.shape[-1],
                                                 new_height=resize[0],
                                                 new_width=resize[1])
        
        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type= device, dtype= torch.float16):
                y_preds = model(X)
                loss, metrics = loss_fn(y_preds, y)
        else:
            y_preds = model(X)
            loss, metrics = loss_fn(y_preds, y)
            
        with torch.no_grad():
            epe = metrics["epe"]
            px_1 = metrics["1px"]
            px_3 = metrics["3px"]
            px_5 = metrics["5px"]
        
        train_l1_loss += loss.item()
        train_epe += epe
        train_1px += px_1
        train_3px += px_3
        train_5px += px_5
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters= model.parameters(),
                                           max_norm= 1.0)
            optimizer.step()
            
        if scheduler:
            scheduler.step()

    train_l1_loss /= len(dataloader)
    train_epe /= len(dataloader)
    train_1px /= len(dataloader)
    train_3px /= len(dataloader)
    train_5px /= len(dataloader)
    
    print(f" train Loss: {train_l1_loss:.5f} | train 1px: {train_1px:.5f} | train 3px: {train_3px:.5f} | train 5px: {train_5px:.5f} | train EPE: {train_epe: .5f}")

    return train_l1_loss, train_epe, train_1px, train_3px, train_5px

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              resize: Tuple[int, int],
              device: torch.device = "cuda"):

    resized = v2.Resize(size=resize)
    normalize = v2.Normalize(mean= [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     std= [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    model.eval()
    test_l1_loss, test_epe, test_1px, test_3px, test_5px = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.inference_mode():
        with torch.autocast(device_type = device, dtype= torch.float16):
            for batch, X in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
                
                img_1, img_2, flow = X
                X, y = torch.cat((img_1, img_2), dim= 1), flow.permute(0, 2, 3, 1)
                X, y = X.to(device), y.to(device)
                X, y = normalize(resized(X)), scale_flow(flow=y,
                                                         original_height = y.shape[-2],
                                                         original_width = y.shape[-1],
                                                         new_height=resize[0],
                                                         new_width=resize[1])
                
                y_preds = model(X)
        
                loss, metrics = loss_fn(y_preds, y)
                epe = metrics["epe"]
                px_1 = metrics["1px"]
                px_3 = metrics["3px"]
                px_5 = metrics["5px"]
                
                test_l1_loss += loss.item()
                test_epe += epe
                test_1px += px_1
                test_3px += px_3
                test_5px += px_5
        
            test_l1_loss /= len(dataloader)
            test_epe /= len(dataloader)
            test_1px /= len(dataloader)
            test_3px /= len(dataloader)
            test_5px /= len(dataloader)
    
    print(f" test Loss: {test_l1_loss:.5f} | test 1px: {test_1px:.5f} | test 3px: {test_3px:.5f} | test 5px: {test_5px:.5f} | test EPE: {test_epe: .5f}")

    return test_l1_loss, test_epe, test_1px, test_3px, test_5px

def train_Sintel(model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 resize: Tuple[int, int],
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 use_scaler: bool = False,
                 device: torch.device = "cuda",
                 epochs: int = 5):

    results = {"train_l1": [],
               "train_epe": [],
               "train_1px": [],
               "train_3px": [],
               "train_5px": [],
               "test_l1": [],
               "test_epe": [],
               "test_1px": [],
               "test_3px": [],
               "test_5px": []}

    model.to(device)

    if use_scaler:
        scaler = GradScaler()
    else:
        scaler = None
        
    for epoch in tqdm(range(epochs)):
        train_l1_loss, train_epe, train_1px, train_3px, train_5px = train_step(model = model,
                                                                               dataloader= train_dataloader,
                                                                               loss_fn= loss_fn,
                                                                               optimizer= optimizer,
                                                                               scheduler= scheduler,
                                                                               resize= resize,
                                                                               device= device,
                                                                               scaler= scaler)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model = model,
                            optimizer = optimizer,
                            checkpoint_name = f"RAFT-Sintel-{epoch + 1}epochs-Experiment3.pth")

        test_l1_loss, test_epe, test_1px, test_3px, test_5px = test_step(model = model,
                                                                         dataloader= test_dataloader,
                                                                         loss_fn= loss_fn,
                                                                         resize= resize,
                                                                         device= device)

        results["train_l1"].append(train_l1_loss)
        results["train_epe"].append(train_epe)
        results["train_1px"].append(train_1px)
        results["train_3px"].append(train_3px)
        results["train_5px"].append(train_5px)
        results["test_l1"].append(test_l1_loss)
        results["test_epe"].append(test_epe)
        results["test_1px"].append(test_1px)
        results["test_3px"].append(test_3px)
        results["test_5px"].append(test_5px)

    return results
