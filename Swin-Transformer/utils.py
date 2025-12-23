import torch
from pathlib import Path

def save_checkpoint(model: torch.nn.Module,
                    checkpoint_name: str,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler = None):

    assert checkpoint_name.endswith(".pth") or checkpoint_name.endswith(".pt"), "Checkpoint name should end with '.pt' or '.pth'"
    
    model_path = Path("models")
    optimizer_path = Path("optimizers")
    scheduler_path = Path("schedulers")

    model_path.mkdir(parents=True, exist_ok=True)
    optimizer_path.mkdir(parents=True, exist_ok=True)
    scheduler_path.mkdir(parents=True, exist_ok=True)

    torch.save(obj = model.state_dict(), f= model_path / checkpoint_name)
    if optimizer:
        torch.save(obj = optimizer.state_dict(), f= optimizer_path / checkpoint_name)
    if scheduler:
        torch.save(obj = scheduler.state_dict(), f= scheduler_path / checkpoint_name)

    print("Model, optimizer, and scheduler are saved")

def load_checkpoint(model: torch.nn.Module,
                    checkpoint_name: str,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                    devce: torch.device = "cuda"):

    assert checkpoint_name.endswith(".pt") or checkpoint_name.endswith(".pth"), "Checkpoint name should end with '.pt' or '.pth'"

    model_path = Path("models")
    optimizer_path = Path("optimizers")
    scheduler_path = Path("schedulers")
    
    model.load_state_dict(torch.load(model_path / checkpoint_name))
    if optimizer:
        optimizer.load_state_dict(torch.load(optimizer_path / checkpoint_name))
    if scheduler:
        scheduler.load_state_dict(torch.load(scheduler_path / checkpoint_name))

def plot_metrics(results: Dict[str, List[float]]):

    epochs = range(len(results["train_loss"]))
    plt.figure(figsize= (12, 30))
    for i, (name, metric) in enumerate(results.items()):
        plt.subplot(len(results), 1, i+1)
        plt.plot(epochs, metric)
        plt.xlabel("epochs")
        plt.ylabel(name.split("_")[1])
        plt.title(f"{name.split('_')[0]} {name.split('_')[1]} per epoch")

    plt.show()
