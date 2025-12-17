import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple

def init_weights(model: torch.nn.Module):
    if isinstance(model, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)

    elif isinstance(model, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int):
        
        super().__init__()

        self.downsample_1 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels= in_channels,
                                    out_channels= 64,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2)),
                ("bn", nn.BatchNorm2d(num_features=64)),
                ("relu", nn.ReLU())
            ])
        )

        self.downsample_2 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels= 64,
                                    out_channels= 128,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2)),
                ("bn", nn.BatchNorm2d(num_features=128)),
                ("relu", nn.ReLU())
            ])
        )

        self.downsample_3 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels= 128,
                                    out_channels= 256,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2)),
                ("bn", nn.BatchNorm2d(num_features=256)),
                ("relu", nn.ReLU())
            ])
        )

        self.fully_connected = nn.Sequential(
            OrderedDict([
                ("flatten", nn.Flatten(start_dim=1, 
                                       end_dim=3)),
                ("linear", nn.Linear(in_features=256 * 8 * 8,
                                     out_features=2048)),
                ("bn", nn.BatchNorm1d(num_features=2048)),
                ("relu", nn.ReLU())
            ])
        )
        self.dense_mean = nn.Sequential(
            OrderedDict([
            ("linear", nn.Linear(in_features=2048,
                                 out_features=latent_dim))
            ])
        )

        self.dense_logvar = nn.Sequential(
            OrderedDict([
            ("linear", nn.Linear(in_features=2048,
                                 out_features=latent_dim))
            ])
        )

    def forward(self,
                x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.downsample_1(x)
        # print(x.shape)
        x = self.downsample_2(x)
        # print(x.shape)
        x = self.downsample_3(x)
        # print(x.shape)
        x = self.fully_connected(x)
        # print(x.shape)
        mean = self.dense_mean(x)
        logvar = self.dense_logvar(x)
        # print(logvar.shape)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 out_channels: int):
        
        super().__init__()

        self.fully_connected = nn.Sequential(
            OrderedDict([
                ("linear", nn.Linear(in_features=latent_dim,
                                     out_features= 8 * 8 * 256)),
                ("bn", nn.BatchNorm1d(num_features= 8 * 8 * 256)),
                ("relu", nn.ReLU())
            ])
        )

        self.upsample_1 = nn.Sequential(
            OrderedDict([
                ("backconv", nn.ConvTranspose2d(in_channels= 256,
                                                out_channels=256,
                                                kernel_size= 5,
                                                stride= 2,
                                                padding=2,
                                               output_padding= 1)),
                ("bn", nn.BatchNorm2d(num_features= 256)),
                ("relu", nn.ReLU())
            ])
        )

        self.upsample_2 = nn.Sequential(
            OrderedDict([
                ("backconv", nn.ConvTranspose2d(in_channels= 256,
                                                out_channels=128,
                                                kernel_size= 5,
                                                stride= 2,
                                                padding=2,
                                               output_padding=1)),
                ("bn", nn.BatchNorm2d(num_features= 128)),
                ("relu", nn.ReLU())
            ])
        )

        self.upsample_3 = nn.Sequential(
            OrderedDict([
                ("backconv", nn.ConvTranspose2d(in_channels= 128,
                                                out_channels=32,
                                                kernel_size= 5,
                                                stride= 2,
                                                padding=2,
                                               output_padding= 1)),
                ("bn", nn.BatchNorm2d(num_features= 32)),
                ("relu", nn.ReLU())
            ])
        )

        self.recontruction = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels= 32,
                                    out_channels=3,
                                    kernel_size= 5,
                                    stride= 1,
                                    padding=2)),
                ("tanh", nn.Tanh())
            ])
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = self.fully_connected(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], 256, 8, 8)
        # print(x.shape)
        x = self.upsample_1(x)
        # print(x.shape)
        x = self.upsample_2(x)
        # print(x.shape)
        x = self.upsample_3(x)
        # print(x.shape)
        x = self.recontruction(x)
        # print(x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int):
        super().__init__()

        self.conv_1 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels= in_channels,
                                    out_channels=32,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)),
                ("bn", nn.BatchNorm2d(num_features= 32)),
                ("relu", nn.ReLU())
            ])
        )

        self.downsample_1 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels= 32,
                                    out_channels=128,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2)),
                ("bn", nn.BatchNorm2d(num_features= 128)),
                ("relu", nn.ReLU())
            ])
        )

        self.downsample_2 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels= 128,
                                    out_channels=256,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2)),
                ("bn", nn.BatchNorm2d(num_features= 256)),
                ("relu", nn.ReLU())
            ])
        )

        self.downsample_3 = nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels= 256,
                                    out_channels=256,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2)),
                ("bn", nn.BatchNorm2d(num_features= 256)),
                ("relu", nn.ReLU())
            ])
        )

        self.fully_connected_1 = nn.Sequential(
            OrderedDict([
                ("flatten", nn.Flatten(start_dim=1,
                                       end_dim=3)),
                ("linear", nn.Linear(in_features=8 * 8 * 256,
                                     out_features=512)),
                ("bn", nn.BatchNorm1d(num_features=512)),
                ("relu", nn.ReLU())
            ])
        )

        self.fully_connected_2 = nn.Sequential(
            OrderedDict([
                ("linear", nn.Linear(in_features=512,
                                     out_features=1)),
                ("sigmoid", nn.Sigmoid())
            ])
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        # print(x.shape)
        x = self.downsample_1(x)
        # print(x.shape)
        x = self.downsample_2(x)
        # print(x.shape)
        x = self.downsample_3(x)
        # print(x.shape)
        disl_out = x.view(-1, 256 * 8 * 8)
        x = self.fully_connected_1(x)
        # print(x.shape)
        x = self.fully_connected_2(x)
        # print(x.shape)
        return disl_out, x

class VAEGAN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 out_channels: int):
        
        super().__init__()

        self.encoder = Encoder(in_channels= in_channels,
                               latent_dim= latent_dim).apply(init_weights)

        self.decoder = Decoder(latent_dim= latent_dim,
                               out_channels= out_channels).apply(init_weights)

        self.discriminator = Discriminator(in_channels= in_channels).apply(init_weights)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        z_mean, z_logvar = self.encoder(x)
        z_std = torch.exp(0.5 * z_logvar)
        p_z = torch.randn_like(z_std)
        z = z_mean + p_z * z_std
        x_tilda = self.decoder(z)

        return z_mean, z_logvar, z, x_tilda
