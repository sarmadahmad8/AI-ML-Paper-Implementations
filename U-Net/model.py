
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DoubleDownSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        
        super().__init__()
        
        self.downsample_block = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=0),
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=out_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=0),
                                              nn.ReLU()
                                             )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.downsample_block(x)
        # print(x.shape)
        return x


class DoubleUpSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):

        super().__init__()

        self.Upsample = nn.Sequential(nn.Upsample(scale_factor=2,
                                                  mode='bilinear'),
                                      nn.Conv2d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)
                                     )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.Upsample(x)
        # print(x.shape)
        return x
        
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        
        super().__init__()

        self.feature_expansion = [64, 128, 256, 512]
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.resample = nn.ModuleList()

        self.double_downsample = nn.Conv2d(in_channels= self.feature_expansion[-1]*2,
                                           out_channels= self.feature_expansion[-1]*2,
                                           kernel_size=3,
                                           stride=1,
                                           padding=0)
        
        for feature in self.feature_expansion:
            self.downsample.append(DoubleDownSample(in_channels=self.in_channels,
                                                    out_channels=feature))
            self.in_channels = feature

        for feature in reversed(self.feature_expansion):
            self.upsample.append(DoubleUpSample(in_channels=feature*2,
                                                out_channels= feature))

        for feature in reversed(self.feature_expansion):
            self.resample.append(DoubleDownSample(in_channels=feature*2,
                                                  out_channels=feature))

        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2)
        
        self.bottleneck = nn.Conv2d(in_channels=self.feature_expansion[-1],
                                    out_channels=self.feature_expansion[-1]*2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=0)
        self.final_conv = nn.Conv2d(in_channels=self.feature_expansion[0],
                                    out_channels=self.out_channels,
                                    kernel_size=1,
                                   stride=1,
                                   padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_connections = []
        for down in self.downsample:
            x =  down(x)
            residual_connections.append(x)
            x = self.maxpool(x)

        x = self.bottleneck(x)
        x = self.double_downsample(x)
        # print(x.shape)
        residual_connections = residual_connections[::-1]
        for i, up in enumerate(self.upsample):
            x = up(x)
            center_crop = transforms.CenterCrop(size=(x.shape[2], x.shape[3]))
            x = torch.cat((x, center_crop(residual_connections[i])), dim=1)
            # print(x.shape)
            x = self.resample[i](x)
            # print(x.shape)

        return self.final_conv(x)
    
