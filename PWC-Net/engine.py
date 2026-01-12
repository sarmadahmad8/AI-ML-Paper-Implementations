import torch
from torch import autocast, GradScaler
from torchvision.transforms import v2
import torch.nn.functional as F
from typing import Tuple
from tqdm.auto import tqdm
from utils import save_checkpoint

def end_point_error(preds: torch.Tensor,
                    targets: torch.tensor):

    epe = torch.sqrt(((preds - targets) ** 2).sum(dim=1))

    return epe.mean()

def fl_all(preds: torch.Tensor,
           targets: torch.Tensor):

    epe = torch.sqrt(((preds - targets) ** 2).sum(dim=1))
    mag = torch.sqrt((targets ** 2).sum(dim = 1))

    outliers = (epe > 3) & ((epe / (mag + 1e-6)) > 0.05)

    return (outliers.float().mean() * 100).item()

def angle_error(preds: torch.Tensor,
                targets: torch.Tensor):

    pred_norm = torch.cat([preds, torch.ones_like(preds[:, :1])], dim= 1)
    target_norm = torch.cat([targets, torch.ones_like(targets[:, :1])], dim= 1)

    dot_product = (pred_norm * target_norm).sum(dim=1)
    pred_mag = torch.sqrt((pred_norm ** 2).sum(dim = 1))
    target_mag = torch.sqrt((target_norm ** 2).sum(dim = 1))

    angle = torch.acos(torch.clamp(dot_product / (pred_mag * target_mag), -1, 1))
    return torch.rad2deg(angle).mean().item()

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               resize: Tuple[int, int],
               scheduler: torch.optim.lr_scheduler.LRScheduler = None,
               scaler: torch.amp.GradScaler = None,
               device: torch.device = "cuda"):

    resized = v2.Resize(size=resize)
    normalize = v2.Normalize(mean= [0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                     std= [0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
    model.train()
    train_l2_loss, train_fl_all, train_aae, train_epe = 0.0, 0.0, 0.0, 0.0

    for batch, X in tqdm(enumerate(dataloader)):
        
        img_1, img_2, flow = X
        X, y = torch.cat((img_1, img_2), dim= 1), flow.permute(0, 2, 3, 1)
        X, y = X.to(device), y.to(device)
        X, y = normalize(resized(X)), resized(y)
        
        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type= device, dtype= torch.float16):
                y_preds = model(X)
                loss = loss_fn(y_preds, y, model.parameters())
        else:
            y_preds = model(X)
            loss = loss_fn(y_preds, y, model.parameters())
            
        with torch.no_grad():
            flow_downsampled = y_preds[0] * 20.0
            pred_flow = F.interpolate(flow_downsampled, 
                                      scale_factor=4, 
                                      mode='bilinear', 
                                      align_corners=True) 
            aae = angle_error(pred_flow, y)
            fl = fl_all(pred_flow, y)
            epe = end_point_error(pred_flow, y)
        
        train_l2_loss += loss
        train_aae += aae
        train_fl_all += fl
        train_epe += epe
        
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
    train_aae /= len(dataloader)
    train_fl_all /= len(dataloader)
    train_epe /= len(dataloader)

    print(f" Train Loss: {train_l2_loss:.5f} | Train AAE: {train_aae:.5f} | Train Fl-all: {train_fl_all:.5f} | Train EPE: {train_epe: .5f}")

    return train_l2_loss, train_aae, train_fl_all, train_epe

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              resize: Tuple[int, int],
              device: torch.device = "cuda"):

    resize = v2.Resize(size=resize)
    model.eval()
    test_epe_loss, test_aae, test_fl_all = 0.0, 0.0, 0.0

    with torch.inference_mode():
        for batch, X in tqdm(enumerate(dataloader)):
            
            img_1, img_2, flow = X
            X, y = torch.cat((img_1, img_2), dim= 1), flow.permute(0, 2, 3, 1)
            X, y = X.to(device), y.to(device)
            X, y = resized(X), resized(y)
            
            y_preds = model(X)
    
            loss = loss_fn(y_preds, y, model.parameters())
            aae = angle_error(y_preds[0], y)
            fl = fl_all(y_preds[0], y)
            
            test_epe_loss += loss
            test_aae += aae
            test_fl_all += fl
    
        test_epe_loss /= len(dataloader)
        test_aae /= len(dataloader)
        test_fl_all /= len(dataloader)
    
        print(f" Test Loss: {test_epe_loss:.5f} | Test AAE: {test_aae:.5f} | Test Fl-all: {test_fl_all:.5f}")

    return test_epe_loss, test_aae, test_fl_all

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

    results = {"train_l2": [],
               "train_epe": [],
               "train_aae": [],
               "train_fl_all": [],
               "test_epe": [],
               "test_aae": [],
               "test_fl_all": []}

    model.to(device)

    if use_scaler:
        scaler = GradScaler()
    else:
        scaler = None
        
    for epoch in tqdm(range(epochs)):
        train_l2, train_aae, train_fl_all, train_epe = train_step(model = model,
                                                                  dataloader= train_dataloader,
                                                                  loss_fn= loss_fn,
                                                                  optimizer= optimizer,
                                                                  scheduler= scheduler,
                                                                  resize= resize,
                                                                  device= device,
                                                                  scaler= scaler)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model = model,
                            optimizer = optimizer,
                            checkpoint_name = f"PWCNet-Sintel-{epoch + 1}epochs-Experiment3.pth")

        results["train_l2"].append(train_l2)
        results["train_epe"].append(train_epe)
        results["train_aae"].append(train_aae)
        results["train_fl_all"].append(train_fl_all)

    return results
