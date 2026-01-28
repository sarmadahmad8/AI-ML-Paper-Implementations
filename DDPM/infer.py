import os
import torchvision
from torchvision.utils import make_grid
from tqdm.auto import tqdm

def sample(model: torch.nn.Module,
           scheduler: LinearNoiseScheduler,
           num_samples: int,
           in_channels: int,
           img_size: int,
           device: torch.device):

    num_timesteps = 1000
    xt = torch.randn([num_samples, in_channels, img_size, img_size], dtype=torch.float32).to(device)

    for i in tqdm(reversed(range(num_timesteps))):

        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        ims = torch.clamp(x0_pred, min=-1.0, max=1.0)
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=num_samples//10)
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join("images", "samples")):
            os.makedirs(os.path.join("images", "samples"))

        img.save(os.path.join("images", "samples", "x0_{}.png".format(i)))
        img.close()
        
device = "cuda" if torch.cuda.is_available() else "cpu"

unet = Unet(in_channels=3).to(device)
unet.load_state_dict(torch.load("models/UNET-diffuser-CIFAR10-60epochs.pth"))

unet.eval()

scheduler = LinearNoiseScheduler(num_timesteps=1000,
                                 beta_start=0.0001,
                                 beta_end=0.02,
                                 device=device)

with torch.inference_mode():
    sample(model = unet,
           scheduler = scheduler,
           num_samples=100,
           in_channels=3,
           img_size=32,
           device=device)
