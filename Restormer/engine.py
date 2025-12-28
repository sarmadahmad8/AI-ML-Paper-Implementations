from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from utils import save_checkpoint
from tqdm.auto import tqdm
import torch
from torch.amp import autocast, GradScaler
from kornia.color.ycbcr import RgbToYcbcr

ycbcr = RgbToYcbcr()

def test_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               device: torch.device,
               amp: bool = False):
        
    test_loss, test_psnr, test_ssim = 0.0, 0.0, 0.0

    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(test_dataloader)):
            
            X, y = X.to(device), y.to(device)
            # y = X - y
            if amp:
                with autocast(dtype= torch.float16,
                             device_type= device):
                    y_preds = model(X)
                    loss = loss_fn(y_preds, y)
            else:
                y_preds = model(X)
                loss = loss_fn(y_preds, y)

            y_preds = ycbcr(y_preds)
            y = ycbcr(y)
            psnr = peak_signal_noise_ratio(y_preds[:, :1, :, :], y[:, :1, :, :],
                                           data_range= (0.0,1.0))
            ssim = structural_similarity_index_measure(y_preds[:, :1, :, :], y[:, :1, :, :],
                                                       data_range= (0.0,1.0))
            
            test_loss += loss.item()
            test_psnr += psnr.item()
            test_ssim += ssim.item()
    
        test_loss /= len(test_dataloader)
        test_psnr /= len(test_dataloader)
        test_ssim /= len(test_dataloader)

    print(f" Test Loss: {test_loss:.5f} | Test PSNR: {test_psnr:.5f} | Test SSIM: {test_ssim:.5f}")
    
    model.train()
    
    return test_loss, test_psnr, test_ssim

def train(model: torch.nn.Module,
           loss_fn: torch.nn.Module,
           optimizer: torch.optim.Optimizer,
           scheduler: torch.optim.lr_scheduler,
           train_dataloader: torch.utils.data.DataLoader,
           test_dataloader: torch.utils.data.DataLoader,
           device: torch.device,
           experiment_name: str,
           amp: bool = False):

    if amp:
        scaler = GradScaler()
        
    results = {"train_loss": [],
               "train_psnr": [],
               "train_ssim": [],
               "test_loss": [],
               "test_psnr": [],
               "test_ssim": []}

    model.to(device)
    
    train_loss, train_psnr, train_ssim = 0.0, 0.0, 0.0

    model.train()
    for batch, (X, y) in tqdm(enumerate(train_dataloader)):
        
        X, y = X.to(device), y.to(device)
        # y = y - X # remove sharp from blur so model learns residuals
        if amp:
            with autocast(dtype= torch.float16,
                         device_type= device):
                y_preds = model(X)
                loss = loss_fn(y_preds, y)
        else:
            y_preds = model(X)
            loss = loss_fn(y_preds, y)
            
        y_preds = ycbcr(y_preds)
        y = ycbcr(y)
        psnr = peak_signal_noise_ratio(y_preds[:, :1, :, :], y[:, :1, :, :],
                                       data_range= (0.0,1.0))
        ssim = structural_similarity_index_measure(y_preds[:, :1, :, :], y[:, :1, :, :], 
                                                  data_range= (0.0,1.0))
        
        train_loss += loss.item()
        train_psnr += psnr.item()
        train_ssim += ssim.item()
        
        optimizer.zero_grad()
        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters= model.parameters(),
                                       max_norm= 1.0)
        if amp:
            scaler.step(optimizer=optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        if (batch+1) % 1000 == 0:
            
            train_loss /= 1000
            train_psnr /= 1000
            train_ssim /= 1000

            results["train_loss"].append(train_loss)
            results["train_psnr"].append(train_psnr)
            results["train_ssim"].append(train_ssim)
            
            print(f" Train Loss: {train_loss:.5f} | Train PSNR: {train_psnr:.5f} | Train SSIM: {train_ssim:.5f}")

            train_loss, train_psnr, train_ssim = 0.0, 0.0, 0.0
            
            test_loss, test_psnr, test_ssim = test_step(model= model,
                                                        test_dataloader= test_dataloader,
                                                        loss_fn= loss_fn,
                                                        device= device,
                                                        amp = amp)
            
            results["test_loss"].append(test_loss)
            results["test_psnr"].append(test_psnr)
            results["test_ssim"].append(test_ssim)

        if (batch+1) % 5000 == 0:
            save_checkpoint(model= model,
                            optimizer= optimizer,
                            scheduler= scheduler,
                            checkpoint_name= f"Restormer-GoPro-{batch+1}iterations-L1-128-{experiment_name}.pth")

    return results
