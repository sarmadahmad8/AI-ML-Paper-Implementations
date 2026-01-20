import torch
from torch import autocast, GradScaler
from torchvision.transforms import v2
import torch.nn.functional as F
from typing import Tuple
from tqdm.auto import tqdm
from utils import save_checkpoint

def end_point_error(preds: torch.Tensor,
                    targets: torch.tensor,
                    valid_mask: torch.tensor):

    epe = torch.sqrt(((preds - targets) ** 2).sum(dim=1))

    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask > 0
            
        epe = torch.masked_select(epe, valid_mask)
        return epe.mean() if epe.numel() > 0 else torch.tensor(0.0, device= preds.device)

    return epe.mean()

def scale_disparity(disparity: torch.Tensor, 
                   original_width: int, 
                   new_width: int) -> torch.Tensor:
    
    scale_factor = new_width / original_width
    return disparity * scale_factor

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               crop: Tuple[int, int],
               scheduler: torch.optim.lr_scheduler.LRScheduler = None,
               scaler: torch.amp.GradScaler = None,
               device: torch.device = "cuda"):

    cropped = v2.CenterCrop(size=crop)
    normalize = v2.Normalize(mean= [0.485, 0.456, 0.406],
                     std= [0.229, 0.224, 0.225])
    model.train()
    train_l2_loss, train_epe = 0.0, 0.0

    for batch, X in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        
        img_l, img_r, disp, val_mask = X
        B, C, H, W = img_l.shape
        img_l, img_r, y, val_mask = img_l.to(device), img_r.to(device), disp.permute(0, 2, 3, 1).to(device), val_mask.to(device)
        #print(y.shape)
        img_l, img_r, y, val_mask = normalize(cropped(img_l)), normalize(cropped(img_r)), cropped(y), cropped(val_mask)
        y = scale_disparity(y, original_width=W, new_width=crop[1])
        y = torch.clamp(y, 0.0, 192.0)
        
        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type= device, dtype= torch.float16):
                y_preds = model(img_l, img_r)
                loss = 0.5 * loss_fn(y_preds[0], y) + 0.7 * loss_fn(y_preds[1], y) + 1.0 * loss_fn(y_preds[2], y)
        else:
            y_preds = model(img_l, img_r)
            loss = 0.5 * loss_fn(y_preds[0], y) + 0.7 * loss_fn(y_preds[1], y) + 1.0 * loss_fn(y_preds[2], y)
            
        with torch.no_grad():
            epe = end_point_error(y_preds[2], y, valid_mask= val_mask)
        
        train_l2_loss += loss.item()
        train_epe += epe.item()
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters= model.parameters(),
            #                                max_norm= 10.0)
            optimizer.step()
            
        if scheduler:
            scheduler.step()

    train_l2_loss /= len(dataloader)
    train_epe /= len(dataloader)

    print(f" Train Loss: {train_l2_loss:.5f} | Train EPE: {train_epe: .5f}")

    return train_l2_loss, train_epe

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              crop: Tuple[int, int],
              device: torch.device = "cuda"):

    cropped = v2.CenterCrop(size=crop)
    normalize = v2.Normalize(mean= [0.485, 0.456, 0.406],
                     std= [0.229, 0.224, 0.225])
    
    model.eval()
    test_epe, test_l2_loss = 0.0, 0.0

    with torch.inference_mode():
        with torch.autocast(device_type= device, dtype=torch.float16):
            for batch, X in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
                
                img_l, img_r, disp, val_mask = X
                B, C, H, W = img_l.shape
                img_l, img_r, y, val_mask = img_l.to(device), img_r.to(device), disp.permute(0, 2, 3, 1).to(device), val_mask.to(device)
                #print(y.shape)
                img_l, img_r, y, val_mask = normalize(cropped(img_l)), normalize(cropped(img_r)), cropped(y), cropped(val_mask)
                y = scale_disparity(y, original_width=W, new_width=crop[1])
                y = torch.clamp(y, 0.0, 192.0)
                
                y_preds = model(img_l, img_r)
        
                loss = 0.5 * loss_fn(y_preds[0], y) + 0.7 * loss_fn(y_preds[1], y) + 1.0 * loss_fn(y_preds[2], y)
                epe = end_point_error(y_preds[2], y, valid_mask= val_mask)
    
                test_l2_loss += loss.item()
                test_epe += epe.item()
        
            test_l2_loss /= len(dataloader)
            test_epe /= len(dataloader)
        
        print(f" Test Loss: {test_l2_loss:.5f} | Test EPE: {test_epe:.5f}")

    return test_l2_loss, test_epe

def train_Stereo(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                crop: Tuple[int, int],
                scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                use_scaler: bool = False,
                device: torch.device = "cuda",
                epochs: int = 5):

    results = {"train_l2": [],
               "train_epe": [],
               "test_l2": [],
               "test_epe": []}

    model.to(device)

    if use_scaler:
        scaler = GradScaler()
    else:
        scaler = None
        
    for epoch in tqdm(range(epochs)):
        train_l2, train_epe = train_step(model = model,
                                      dataloader= train_dataloader,
                                      loss_fn= loss_fn,
                                      optimizer= optimizer,
                                      scheduler= scheduler,
                                      crop= crop,
                                      device= device,
                                      scaler= scaler)

        results["train_l2"].append(train_l2)
        results["train_epe"].append(train_epe)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model = model,
                            optimizer = optimizer,
                            checkpoint_name = f"PSMNet-Sintel-{epoch + 1 + 40}epochs-Experiment1.pth")

        test_l2, test_epe = test_step(model = model,
                                      dataloader= test_dataloader,
                                      loss_fn= loss_fn,
                                      crop= crop,
                                      device= device)

        results["test_l2"].append(test_l2)
        results["test_epe"].append(test_epe)

    return results
