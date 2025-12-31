import torch
import torch.nn as nn
from data_setup import create_dataloaders_MovingMNIST
from model import ConvLSTM
from engine import train_MovingMNIST
from utils import plot_metrics, plot_reconstructed_sequence, save_checkpoint

EPOCHS = 70
LR = 1e-3
ALPHA = 0.9
WEIGHT = torch.tensor(19.2929, dtype= torch.float32)
print(WEIGHT)
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders_MovingMNIST(batchsize= 64,
                                                                                                 num_workers= 8)

convlstm = ConvLSTM(in_channels= (16, 128, 64, 64, 128),
                      out_channels=1,
                      embed_dim= (128, 64, 64, 128, 64),
                      patch_size= 16,
                      layers=5)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.RMSprop(params = convlstm.parameters(),
                                lr= LR,
                                alpha= ALPHA)

results = train_MovingMNIST(model = convlstm,
                            train_dataloader= train_dataloader,
                            test_dataloader= test_dataloader,
                            loss_fn= loss_fn,
                            optimizer= optimizer,
                            device= device,
                            epochs= EPOCHS)

save_checkpoint(model= convlstm,
                optimizer= optimizer,
                checkpoint_name= f"ConvLSTM-MovingMNIST-{EPOCHS}epochs.pth")

plot_metrics(results= results)

plot_reconstructed_sequence(model = convlstm,
                            dataset= test_dataset,
                            samples = 3)
