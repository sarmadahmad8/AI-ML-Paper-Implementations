import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils import load_checkpoint, save_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = LinearNoiseScheduler(num_timesteps=1000,
                                 beta_start=0.0001,
                                 beta_end=0.02,
                                 device=device)

unet = Unet(in_channels=3).to(device)
unet.compile()

num_epochs = 60
optimizer = torch.optim.Adam(unet.parameters(),
                             lr= 0.0001)

load_checkpoint(model=unet,
                optimizer=optimizer,
                checkpoint_name="UNET-diffuser-CIFAR10-60epochs.pth")

loss_fn = torch.nn.MSELoss()

scaler = torch.GradScaler()

unet.train()
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    losses = []
    for img, _ in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
        img = img.to(device)
        optimizer.zero_grad()

        noise = torch.randn_like(img).to(device)

        t = torch.randint(0, 1000, (img.shape[0],)).to(device)

        noisy_img = scheduler.add_noise(original=img, 
                                        noise=noise,
                                        t=t)
        with torch.autocast(device_type=device, dtype=torch.float16):
            noisy_pred = unet(noisy_img, t)
            loss = loss_fn(noisy_pred, noise)
            
        losses.append(loss.item())
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    total_loss = torch.tensor(losses).sum() / len(losses)
    print(f'Epoch: {epoch} | Loss: {total_loss:.5f} ')

save_checkpoint(model=unet,
                optimizer=optimizer,
                checkpoint_name="UNET-diffuser-CIFAR10-60epochs.pth")
