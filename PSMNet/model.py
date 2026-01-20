import torch
import torch.nn as nn

class ResBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channels: int = 32,
                 dilation: int = 1,
                 stride: int = 1):

        super().__init__()

        self.res_block = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=channels,
                                                 kernel_size=3,
                                                 stride=stride,
                                                 dilation=dilation,
                                                 padding=dilation,
                                                 padding_mode="zeros"),
                                       nn.BatchNorm2d(num_features=channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=channels,
                                                 out_channels=channels,
                                                 kernel_size=3,
                                                 stride=1,
                                                 dilation=dilation,
                                                 padding=dilation,
                                                 padding_mode="zeros"),
                                       nn.ReLU(inplace=True))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.res_block(x)

class ResBlockStack(nn.Module):

    def __init__(self,
                 in_channels: int,
                 blocks: int,
                 channels: int = 32,
                 dilation: int = 1,
                 stride: int = 1):

        super().__init__()

        self.resblock_stack = nn.ModuleList()

        self.resblock_stack.append(ResBlock(in_channels=in_channels,
                                            channels=channels,
                                            stride = stride,
                                            dilation=dilation))
        for b in range(blocks - 1):
            self.resblock_stack.append(ResBlock(in_channels=channels,
                                                channels=channels,
                                                stride=1,
                                                dilation=dilation))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        for block in self.resblock_stack:
            x = block(x)

        return x
                                            
                                                      

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_0 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=32,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=32,
                                              out_channels=32,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=32,
                                              out_channels=32,
                                              kernel_size=3, 
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.ReLU(inplace=True))

        self.conv_1 = ResBlockStack(in_channels=32,
                                    channels=32,
                                    stride=1,
                                    dilation=1,
                                    blocks=3)

        self.conv_2 = ResBlockStack(in_channels=32,
                                    channels=64,
                                    stride=2,
                                    dilation=1,
                                    blocks=16)

        self.conv_3 = ResBlockStack(in_channels=64,
                                    channels=128,
                                    stride=1,
                                    dilation=2,
                                    blocks=3)

        self.conv_4 = ResBlockStack(in_channels=128,
                                    channels=128,
                                    stride=1,
                                    dilation=4,
                                    blocks=3)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = self.conv_0(x)
        x = self.conv_1(x)
        out_2_16 = self.conv_2(x)
        x = self.conv_3(out_2_16)
        out_4_3 = self.conv_4(x)
        # print(out_2_16.shape, out_4_3.shape)
        return out_2_16, out_4_3

class SpatialPyramidPooling(nn.Module):

    def __init__(self,
                 height: int,
                 width: int):

        super().__init__()

        self.branch_1 = nn.Sequential(nn.AvgPool2d(kernel_size=64,
                                                   stride=64),
                                      nn.Conv2d(in_channels=128,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Upsample(size=(height, width),
                                                  mode="bilinear",
                                                  align_corners=False))

        self.branch_2 = nn.Sequential(nn.AvgPool2d(kernel_size=32,
                                                   stride=32),
                                      nn.Conv2d(in_channels=128,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Upsample(size=(height, width),
                                                  mode="bilinear",
                                                  align_corners=False))

        self.branch_3 = nn.Sequential(nn.AvgPool2d(kernel_size=16,
                                                   stride=16),
                                      nn.Conv2d(in_channels=128,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Upsample(size=(height, width),
                                                  mode="bilinear",
                                                  align_corners=False))

        self.branch_4 = nn.Sequential(nn.AvgPool2d(kernel_size=8,
                                                   stride=8),
                                      nn.Conv2d(in_channels=128,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Upsample(size=(height, width),
                                                  mode="bilinear",
                                                  align_corners=False))

        self.fusion = nn.Sequential(nn.Conv2d(in_channels=320,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=128,
                                              out_channels=32,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"))

    def forward(self,
                x_in_2_16: torch.Tensor,
                x_in_4_3: torch.Tensor) -> torch.Tensor:

        x_1 = self.branch_1(x_in_4_3)
        x_2 = self.branch_2(x_in_4_3)
        x_3 = self.branch_3(x_in_4_3)
        x_4 = self.branch_4(x_in_4_3)
        # print(x_1.shape, x_2.shape, x_3.shape, x_4.shape)
        fusion_in = torch.cat([x_1, x_2, x_3, x_4, x_in_4_3, x_in_2_16], dim=1)

        out = self.fusion(fusion_in)

        return out

class PSMNet(nn.Module):

    def __init__(self,
                 height: int,
                 width: int,
                 max_disp: int):

        super().__init__()
        self.height = height
        self.width = width
        self.num_disp_levels = max_disp // 4

        self.feature_extractor = CNN()
        self.spatial_pyramid_pooling = SpatialPyramidPooling(height= height // 4,
                                                             width= width // 4)
        
        self.conv3d_0 = nn.Sequential(nn.Conv3d(in_channels=64,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm3d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=32,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm3d(num_features=32),
                                      nn.ReLU(inplace=True))

        self.conv3d_1 = nn.Sequential(nn.Conv3d(in_channels=32,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm3d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=32,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm3d(num_features=32))

        self.stack3d_1_1 = nn.Sequential(nn.Conv3d(in_channels=32,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True))

        self.stack3d_1_2 = nn.Sequential(nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True))

        self.stack3d_1_3 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
                                                            out_channels=64,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1,
                                                            padding_mode="zeros"),
                                         nn.BatchNorm3d(num_features=64))

        self.stack3d_1_4 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
                                                            out_channels=32,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1,
                                                            padding_mode="zeros"),
                                         nn.BatchNorm3d(num_features=32))

        self.stack3d_2_1 = nn.Sequential(nn.Conv3d(in_channels=32,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True))

        self.stack3d_2_2 = nn.Sequential(nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True))

        self.stack3d_2_3 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
                                                            out_channels=64,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1,
                                                            padding_mode="zeros"),
                                         nn.BatchNorm3d(num_features=64))

        self.stack3d_2_4 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
                                                            out_channels=32,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1,
                                                            padding_mode="zeros"),
                                         nn.BatchNorm3d(num_features=32))

        self.stack3d_3_1 = nn.Sequential(nn.Conv3d(in_channels=32,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True))

        self.stack3d_3_2 = nn.Sequential(nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.BatchNorm3d(num_features=64),
                                          nn.ReLU(inplace=True))

        self.stack3d_3_3 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
                                                            out_channels=64,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1,
                                                            padding_mode="zeros"),
                                         nn.BatchNorm3d(num_features=64))

        self.stack3d_3_4 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,
                                                            out_channels=32,
                                                            kernel_size=3,
                                                            stride=2,
                                                            padding=1,
                                                            output_padding=1,
                                                            padding_mode="zeros"),
                                         nn.BatchNorm3d(num_features=32))

        self.output_0 = nn.Sequential(nn.Conv3d(in_channels=32,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm3d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=32,
                                                out_channels=1,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"))

        self.output_1 = nn.Sequential(nn.Conv3d(in_channels=32,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm3d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=32,
                                                out_channels=1,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"))

        self.output_2 = nn.Sequential(nn.Conv3d(in_channels=32,
                                                out_channels=32,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm3d(num_features=32),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channels=32,
                                                out_channels=1,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"))

        self.upsample = nn.Upsample(size= (max_disp, height, width),
                                    mode="trilinear",
                                    align_corners=False)

        self.softmax = nn.Softmax(dim=1)

    def _cost_volume(self,
                     left_features: torch.Tensor,
                     right_features: torch.Tensor) -> torch.Tensor:

        B, C, H, W = left_features.shape
        D = self.num_disp_levels
        
        cost = torch.zeros(B, 2*C, D, H, W, 
                          dtype=left_features.dtype, 
                          device=left_features.device)
        
        for i in range(D):
            if i > 0:
                cost[:, :C, i, :, i:] = left_features[:, :, :, i:]
                cost[:, C:, i, :, i:] = right_features[:, :, :, :-i]
            else:
                cost[:, :C, i, :, :] = left_features
                cost[:, C:, i, :, :] = right_features
        
        return cost.contiguous()

    def disparity_regression(self,
                             out: torch.Tensor,
                             max_disp: int) -> torch.Tensor:

        disp_indices = torch.arange(0, max_disp, dtype=torch.float32, device= out.device).view(1, max_disp, 1, 1)
        disp = torch.sum(out * disp_indices, dim=1, keepdim=True)

        return disp

    def forward(self,
                left: torch.Tensor,
                right: torch.Tensor) -> torch.Tensor:

        left_feat = self.spatial_pyramid_pooling(*self.feature_extractor(left))
        right_feat = self.spatial_pyramid_pooling(*self.feature_extractor(right))

        cost = self._cost_volume(left_features=left_feat,
                                 right_features=right_feat)

        x = self.conv3d_0(cost)
        x = self.conv3d_1(x) + x
        
        _3d_1_1 = self.stack3d_1_1(x)
        _3d_1_2 = self.stack3d_1_2(_3d_1_1)
        _3d_1_3 = self.stack3d_1_3(_3d_1_2) + _3d_1_1
        _3d_1_4 = self.stack3d_1_4(_3d_1_3) + x

        _3d_2_1 = self.stack3d_2_1(x) + _3d_1_3
        _3d_2_2 = self.stack3d_2_2(_3d_2_1)
        _3d_2_3 = self.stack3d_2_3(_3d_2_2) + _3d_1_1
        _3d_2_4 = self.stack3d_2_4(_3d_2_3) + x

        _3d_3_1 = self.stack3d_3_1(x) + _3d_2_3
        _3d_3_2 = self.stack3d_3_2(_3d_3_1)
        _3d_3_3 = self.stack3d_3_3(_3d_3_2) + _3d_1_1
        _3d_3_4 = self.stack3d_3_4(_3d_3_3) + x

        out_0 = self.output_0(_3d_1_4)
        out_1 = self.output_1(_3d_2_4) + out_0
        out_2 = self.output_2(_3d_3_4) + out_1

        up_out_0 = self.softmax(self.upsample(out_0).squeeze(dim=1))
        up_out_1 = self.softmax(self.upsample(out_1).squeeze(dim=1))
        up_out_2 = self.softmax(self.upsample(out_2).squeeze(dim=1))

        disp_0 = self.disparity_regression(out= up_out_0,
                                           max_disp= up_out_0.shape[1])
        disp_1 = self.disparity_regression(out= up_out_1,
                                           max_disp= up_out_0.shape[1])
        disp_2 = self.disparity_regression(out= up_out_1,
                                           max_disp= up_out_0.shape[1])
        
        #print(disp_0.shape, disp_1.shape, disp_2.shape)
        return disp_0, disp_1, disp_2
