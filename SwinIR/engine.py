2from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm.auto import tqdm
import torch
from kornia.color.ycbcr import RgbToYcbcr

ycbcr = RgbToYcbcr()
def train_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler,
               train_dataloader: torch.utils.data.DataLoader,
               device: torch.device):

    train_loss, train_psnr, train_ssim = 0.0, 0.0, 0.0

    model.train()
    for batch, (X, y) in tqdm(enumerate(train_dataloader)):
        
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        loss = loss_fn(y_preds, y)
        
        y_preds = ycbcr(y_preds)
        y = ycbcr(y)
        psnr = peak_signal_noise_ratio(y_preds[:, :1, :, :], y[:, :1, :, :])
        ssim = structural_similarity_index_measure(y_preds[:, :1, :, :], y[:, :1, :, :])
        
        train_loss += loss.item()
        train_psnr += psnr.item()
        train_ssim += ssim.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss /= len(train_dataloader)
    train_psnr /= len(train_dataloader)
    train_ssim /= len(train_dataloader)

    print(f" Train Loss: {train_loss:.5f} | Train PSNR: {train_psnr:.5f} | Train SSIM: {train_ssim:.5f}")

    return train_loss, train_psnr, train_ssim

def test_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               device: torch.device):

    test_loss, test_psnr, test_ssim = 0.0, 0.0, 0.0

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(test_dataloader)):
            
            X, y = X.to(device), y.to(device)
            y_preds = model(X)
            loss = loss_fn(y_preds, y)

            y_preds = ycbcr(y_preds)
            y = ycbcr(y)
            psnr = peak_signal_noise_ratio(y_preds[:, :1, :, :], y[:, :1, :, :])
            ssim = structural_similarity_index_measure(y_preds[:, :1, :, :], y[:, :1, :, :])
            
            test_loss += loss.item()
            test_psnr += psnr.item()
            test_ssim += ssim.item()
    
        test_loss /= len(test_dataloader)
        test_psnr /= len(test_dataloader)
        test_ssim /= len(test_dataloader)

    print(f" Test Loss: {test_loss:.5f} | Test PSNR: {test_psnr:.5f} | Test SSIM: {test_ssim:.5f}")

    return test_loss, test_psnr, test_ssim

def train(model: torch.nn.Module,
           loss_fn: torch.nn.Module,
           optimizer: torch.optim.Optimizer,
           scheduler: torch.optim.lr_scheduler,
           train_dataloader: torch.utils.data.DataLoader,
           test_dataloader: torch.utils.data.DataLoader,
           device: torch.device,
           epochs: int):

    results = {"train_loss": [],
               "train_psnr": [],
               "train_ssim": [],
               "test_loss": [],
               "test_psnr": [],
               "test_ssim": []}

    model.to(device)
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_psnr, train_ssim = train_step(model= model,
                                                     train_dataloader= train_dataloader,
                                                     loss_fn= loss_fn,
                                                     optimizer= optimizer,
                                                     scheduler= scheduler,
                                                     device= device)
        results["train_loss"].append(train_loss)
        results["train_psnr"].append(train_psnr)
        results["train_ssim"].append(train_ssim)

        test_loss, test_psnr, test_ssim = test_step(model= model,
                                                    test_dataloader= test_dataloader,
                                                    loss_fn= loss_fn,
                                                    device= device)

        results["test_loss"].append(test_loss)
        results["test_psnr"].append(test_psnr)
        results["test_ssim"].append(test_ssim)

    return results
