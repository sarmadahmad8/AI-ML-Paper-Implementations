import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class PerceptualLoss(nn.Module):
    def __init__(self):

        super().__init__()
        
        vgg_weights = torchvision.models.VGG16_Weights.DEFAULT
        vgg = torchvision.models.vgg16(weights= vgg_weights).features

        self.feature_extractor = nn.Sequential(*list(vgg[:16]))
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor):

        preds_features = self.feature_extractor(preds)
        target_features = self.feature_extractor(targets)
        loss = F.l1_loss(preds_features, target_features)
        
        return loss

class L1withPerceptualLoss(nn.Module):

    def __init__(self,
                perceptual_weight: float = 0.1):

        super().__init__()
        
        self.perceptual_weight = perceptual_weight

        self.perceptual_loss = PerceptualLoss()

        self.l1_loss = nn.L1Loss()

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor):

        l1_loss_output = self.l1_loss(preds, targets)
        perceptual_loss_output = self.perceptual_loss(preds, targets)
        total_loss = l1_loss_output + self.perceptual_weight * perceptual_loss_output

        return total_loss


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
        x = self.downsample_1(x)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        x = self.fully_connected_1(x)
        x = self.fully_connected_2(x)
        return x

class L1withPerceptualandGANLoss(nn.Module):

    def __init__(self,
                 batch_size: int,
                 perceptual_weight: float = 0.1,
                 gan_weight: float = 0.1,
                 l1_weight: float = 0.8):

        super().__init__()

        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        self.l1_weight = l1_weight
        
        self.perceptual_loss = PerceptualLoss()
        self.discriminator = Discriminator(in_channels= 3)
        self.l1_loss = nn.L1Loss()
        self.gan_loss = nn.BCELoss()

        self.ones_label = torch.ones([batch_size, 1], 
                                     requires_grad=False)

        self.zeros_label = torch.zeros([batch_size, 1],
                                       requires_grad=False)

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor):

        perceptual_loss_output = self.perceptual_loss(preds, targets)
        l1_loss_output = self.l1_loss(preds, targets)
        gan_loss_output = self.gan_loss(preds, self.zeros_label) + self.gan_loss(targets, self.ones_label)

        total_loss = self.l1_weight * l1_loss_output + self.perceptual_weight * perceptual_loss_output + self.gan_weight * gan_loss_output

        return total_loss
