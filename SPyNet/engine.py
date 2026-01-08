import torch
from torchvision.transforms import v2
from loss import EndPointErrorLoss
from typing import Tuple
from tqdm.auto import tqdm

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
               device: torch.device = "cuda"):

    resize = v2.Resize(size=resize)
    normalize = v2.Normalize(mean= [0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                     std= [0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
    model.train()
    train_epe_loss, train_fl_all, train_aae = 0.0, 0.0, 0.0

    for batch, X in tqdm(enumerate(dataloader)):
        
        img_1, img_2, flow = X
        # print(flow.shape)
        X, y = torch.cat((img_1, img_2), dim= 1), flow.permute(0, 2, 3, 1)
        X, y = X.to(device), y.to(device)
        X, y = normalize(resize(X)), resize(y)

        y_preds = model(X)
        # print(y_preds.shape, y.shape)
        loss = loss_fn(y_preds, y)
        with torch.no_grad():
            aae = angle_error(y_preds, y)
            fl = fl_all(y_preds, y)
        
        train_epe_loss += loss
        train_aae += aae
        train_fl_all += fl
        
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters= model.parameters(),
        #                                max_norm= 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

    train_epe_loss /= len(dataloader)
    train_aae /= len(dataloader)
    train_fl_all /= len(dataloader)

    print(f" Train Loss: {train_epe_loss:.5f} | Train AAE: {train_aae:.5f} | Train Fl-all: {train_fl_all:.5f}")

    return train_epe_loss, train_aae, train_fl_all

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
            X, y = resize(X), resize(y)
    
            y_preds = model(X)
    
            loss = loss_fn(y_preds, y)
            with torch.no_grad():
                aae = angle_error(y_preds, y)
                fl = fl_all(y_preds, y)
            
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
                 device: torch.device = "cuda",
                 epochs: int = 5):

    results = {"train_epe": [],
               "train_aae": [],
               "train_fl_all": [],
               "test_epe": [],
               "test_aae": [],
               "test_fl_all": []}

    model.to(device)
    for epoch in tqdm(range(epochs)):
        train_epe, train_aae, train_fl_all = train_step(model = model,
                                                        dataloader= train_dataloader,
                                                        loss_fn= loss_fn,
                                                        optimizer= optimizer,
                                                        scheduler= scheduler,
                                                        resize= resize,
                                                        device= device)


        results["train_epe"].append(train_epe)
        results["train_aae"].append(train_aae)
        results["train_fl_all"].append(train_fl_all)

    return results
