import torch
import torch.nn as nn
from collections import OrderedDict

class Downsample(nn.Module):
    def __init__(self,
                 in_channels: int):
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            OrderedDict([
                ("conv_1",nn.Conv2d(in_channels=in_channels,
                                                    out_channels= 32,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1)),
                 ("bn_1", nn.BatchNorm2d(num_features=32)),
                 ("relu_1", nn.ReLU())
            ])
        )
        self.ds_conv_layer_1 = nn.Sequential(
                    OrderedDict([
                        ("depthwise_1",nn.Conv2d(in_channels=32,
                                                        out_channels=32,
                                                        kernel_size=3,
                                                        stride= 2,
                                                        padding=1,
                                                          groups=32)),
                        ("pointwise_1",nn.Conv2d(in_channels=32,
                                                        out_channels=48,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0)),
                        ("bn_2",nn.BatchNorm2d(num_features=48)),
                        ("relu_2",nn.ReLU())
                    ])
        )

        self.ds_conv_layer_2 = nn.Sequential(
            OrderedDict([
                ("depthwise_2",nn.Conv2d(in_channels= 48,
                                        out_channels=48,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)),
                ("pointwise_2",nn.Conv2d(in_channels=48,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)),
                ("bn_3",nn.BatchNorm2d(num_features=64)),
                ("relu_3",nn.ReLU())
            ])
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.ds_conv_layer_2(self.ds_conv_layer_1(self.conv_layer_1(x)))

class Bottleneck(nn.Module):
    def __init__(self,
                 expansion_factor: int,
                 stride: int,
                 in_channels: int,
                 out_channels: int):
        super().__init__()

        self.bottleneck = nn.Sequential(
            OrderedDict([
                ("pointwise_3",nn.Conv2d(in_channels= in_channels,
                                        out_channels= in_channels * expansion_factor,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)),
                ("bn_4", nn.BatchNorm2d(num_features= in_channels * expansion_factor)),
                ("relu_4",nn.ReLU()),
                ("depthwise_3",nn.Conv2d(in_channels= in_channels * expansion_factor,
                                        out_channels= in_channels * expansion_factor,
                                        kernel_size=3,
                                        stride= stride,
                                        padding=1,
                                         groups=in_channels * expansion_factor)),
                ("bn_5", nn.BatchNorm2d(num_features= in_channels * expansion_factor)),
                ("relu_5",nn.ReLU()),
                ("pointwise_4",nn.Conv2d(in_channels= in_channels * expansion_factor,
                                        out_channels= out_channels,
                                        kernel_size= 1,
                                        stride=1,
                                        padding=0))
            ])
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.bottleneck(x)

class PyramidPoolingModule(nn.Module):
    def __init__(self,
                in_channels: int,
                channel_reduction: int):
        super().__init__()

        self.pyramid_1 = nn.Sequential(
            OrderedDict([
                ("avepool_1",nn.AdaptiveAvgPool2d(output_size=(6, 6*2))),
                ("pointwise_5",nn.Conv2d(in_channels=in_channels,
                                        out_channels= channel_reduction,
                                        kernel_size= 1,
                                        stride=1,
                                        padding=0)),
                ("upsample_1",nn.Upsample(size=(32, 64),
                                        mode= "bilinear"))
            ])
        )
        self.pyramid_2 = nn.Sequential(
            OrderedDict([
                ("avepool_2",nn.AdaptiveAvgPool2d(output_size=(3, 3*2))),
                ("pointwise_6",nn.Conv2d(in_channels=in_channels,
                                        out_channels= channel_reduction,
                                        kernel_size= 1,
                                        stride=1,
                                        padding=0)),
                ("upsample_2",nn.Upsample(size=(32, 64),
                                        mode= "bilinear"))
            ])
        )
        self.pyramid_3 = nn.Sequential(
            OrderedDict([
                ("avepool_3",nn.AdaptiveAvgPool2d(output_size=(2, 2*2))),
                ("pointwise_7",nn.Conv2d(in_channels=in_channels,
                                        out_channels= channel_reduction,
                                        kernel_size= 1,
                                        stride=1,
                                        padding=0)),
                ("upsample_3",nn.Upsample(size=(32, 64),
                                        mode= "bilinear"))
            ])
        )
        self.pyramid_4 = nn.Sequential(
            OrderedDict([
                ("avepool_4", nn.AdaptiveAvgPool2d(output_size=(1, 1*2))),
                ("pointwise_8", nn.Conv2d(in_channels=in_channels,
                                        out_channels= channel_reduction,
                                        kernel_size= 1,
                                        stride=1,
                                        padding=0)),
                ("upsample_4", nn.Upsample(size=(32, 64),
                                         mode= "bilinear"))
            ])
        )
        

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        p1 = self.pyramid_1(x)
        # print(p1.shape)
        p2 = self.pyramid_2(x)
        # print(p2.shape)
        p3 = self.pyramid_3(x)
        # print(p3.shape)
        p4 = self.pyramid_4(x)
        # print(p4.shape)
        ppm = torch.cat((p1, p2, p3, p4), dim=1)
        # print(ppm.shape)
        
        return ppm

class FeatureFusionModule(nn.Module):
    def __init__(self,
                 in_channels: int):

        super().__init__()

        self.ffm = nn.Sequential(
            OrderedDict([
                ("upsample_5", nn.Upsample(size=(128, 256), 
                                        mode= "bilinear")),
                ("depthwise_4", nn.Conv2d(in_channels=in_channels,
                                        out_channels= in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=4,
                                       dilation= 4,
                                        groups= in_channels)),
                ("bn_6", nn.BatchNorm2d(num_features=in_channels)),
                ("relu_6", nn.ReLU()),
                ("pointwise_9", nn.Conv2d(in_channels= in_channels,
                                        out_channels= in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0))
            ])
        )
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.ffm(x)

class Classifier(nn.Module):
    def __init__(self,
                 in_channels: int,
                out_channels: int,
                dropout: float = 0.1):

        super().__init__()

        self.classifier = nn.Sequential(
            OrderedDict([
                ("depthwise_5", nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups= in_channels)),
                ("pointwise_10", nn.Conv2d(in_channels=in_channels,
                                        out_channels= in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)),
                ("depthwise_6", nn.Conv2d(in_channels= in_channels,
                                        out_channels= in_channels,
                                        kernel_size= 3,
                                        stride=1,
                                        padding=1,
                                        groups= in_channels)),
                ("pointwise_11", nn.Conv2d(in_channels= in_channels,
                                        out_channels= in_channels,
                                        kernel_size= 1,
                                        stride= 1,
                                        padding=0)),
                ("pointwise_12", nn.Conv2d(in_channels= in_channels,
                                        out_channels= out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)),
                ("dropout_1", nn.Dropout2d(p=dropout,
                                         inplace=True))
            ])
        )

    def forward(self,
                x:torch.Tensor) -> torch.Tensor:

        return self.classifier(x)


class FastSCNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion_factor: int):

        super().__init__()

        self.downsampler = Downsample(in_channels= in_channels)

        self.bottleneck_1 = nn.Sequential(Bottleneck(in_channels= 64,
                                                    out_channels= 64,
                                                     expansion_factor= expansion_factor,
                                                    stride= 2),
                                          Bottleneck(in_channels= 64,
                                                    out_channels= 64,
                                                     expansion_factor= expansion_factor,
                                                    stride= 1),
                                          Bottleneck(in_channels= 64,
                                                    out_channels= 64,
                                                     expansion_factor= expansion_factor,
                                                    stride= 1)
                                         )
        self.bottleneck_2 = nn.Sequential(Bottleneck(in_channels= 64,
                                                    out_channels= 96,
                                                     expansion_factor= expansion_factor,
                                                    stride= 2),
                                          Bottleneck(in_channels= 96,
                                                    out_channels= 96,
                                                     expansion_factor= expansion_factor,
                                                    stride= 1),
                                          Bottleneck(in_channels= 96,
                                                    out_channels= 96,
                                                     expansion_factor= expansion_factor,
                                                    stride= 1)
                                         )
        self.bottleneck_3 = nn.Sequential(Bottleneck(in_channels= 96,
                                                    out_channels= 128,
                                                     expansion_factor= expansion_factor,
                                                    stride= 2),
                                          Bottleneck(in_channels= 128,
                                                    out_channels= 128,
                                                     expansion_factor= expansion_factor,
                                                    stride= 1),
                                          Bottleneck(in_channels= 128,
                                                    out_channels= 128,
                                                     expansion_factor= expansion_factor,
                                                    stride= 1)
                                         )
        self.ppm = PyramidPoolingModule(in_channels= 128,
                                        channel_reduction= 128 // 4)


        self.ffm = FeatureFusionModule(in_channels= 128)

        self.conv_upsample = nn.Conv2d(in_channels= 64,
                                       out_channels= 128,
                                       kernel_size= 1,
                                       stride= 1,
                                       padding=0)
        self.relu = nn.ReLU()

        self.classifier = Classifier(in_channels= 128,
                                     out_channels= out_channels)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        down_sample = self.downsampler(x)
        # print(down_sample.shape)
        bottleneck_block = self.bottleneck_3(self.bottleneck_2(self.bottleneck_1(down_sample)))
        global_feature_extractor = self.ppm(bottleneck_block)
        feature_fusion = self.relu(self.conv_upsample(down_sample) + self.ffm(global_feature_extractor))
        classifier = self.classifier(feature_fusion)
        return classifier
        
