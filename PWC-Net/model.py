import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple

class FeaturePyramidExtractorNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                             out_channels=16,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             padding_mode="zeros"),
                                   nn.LeakyReLU(negative_slope=0.1,
                                                inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                             out_channels=32,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             padding_mode="zeros"),
                                   nn.LeakyReLU(negative_slope=0.1,
                                                inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                             out_channels=64,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             padding_mode="zeros"),
                                   nn.LeakyReLU(negative_slope=0.1,
                                                inplace=False))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64,
                                             out_channels=96,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             padding_mode="zeros"),
                                   nn.LeakyReLU(negative_slope=0.1,
                                                inplace=False))

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=96,
                                             out_channels=128,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             padding_mode="zeros"),
                                   nn.LeakyReLU(negative_slope=0.1,
                                                inplace=False))

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=128,
                                             out_channels=192,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             padding_mode="zeros"),
                                   nn.LeakyReLU(negative_slope=0.1,
                                                inplace=False))
                
    def forward(self,
                I_t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            
        c1 = self.conv1(I_t)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        return c6, c5, c4, c3, c2

class OpticalFLowEstimator(nn.Module):

    def __init__(self,
                in_channels: int):

        super().__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=128,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           padding_mode="zeros"),
                                 nn.LeakyReLU(negative_slope=0.1,
                                              inplace=False))
        
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=in_channels + 128,
                                           out_channels=96,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           padding_mode="zeros"),
                                 nn.LeakyReLU(negative_slope=0.1,
                                              inplace=False))
        
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=in_channels + 96,
                                           out_channels=64,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           padding_mode="zeros"),
                                 nn.LeakyReLU(negative_slope=0.1,
                                              inplace=False))
        
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=in_channels + 64,
                                           out_channels=32,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           padding_mode="zeros"),
                                 nn.LeakyReLU(negative_slope=0.1,
                                              inplace=False))
        
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels=in_channels + 32,
                                           out_channels=2,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           padding_mode="zeros"))

    def forward(self,
                c_v_2: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(c_v_2)
        x = self.conv_2(torch.cat((c_v_2, x), dim = 1))
        x = self.conv_3(torch.cat((c_v_2, x), dim = 1))
        x = self.conv_4(torch.cat((c_v_2, x), dim = 1))
        x = self.conv_5(torch.cat((c_v_2, x), dim = 1))
        return x


class ContextNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.context_network = nn.Sequential(nn.Conv2d(in_channels=34,
                                                       out_channels=128,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1,
                                                       padding_mode="zeros",
                                                       dilation=1),
                                               nn.LeakyReLU(negative_slope=0.1,
                                                            inplace=False),
                                               nn.Conv2d(in_channels=128,
                                                       out_channels=128,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=2,
                                                       padding_mode="zeros",
                                                       dilation=2),
                                               nn.LeakyReLU(negative_slope=0.1,
                                                            inplace=False),
                                               nn.Conv2d(in_channels=128,
                                                       out_channels=128,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=4,
                                                       padding_mode="zeros",
                                                       dilation=4),
                                               nn.LeakyReLU(negative_slope=0.1,
                                                            inplace=False),
                                               nn.Conv2d(in_channels=128,
                                                       out_channels=96,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=8,
                                                       padding_mode="zeros",
                                                       dilation=8),
                                               nn.LeakyReLU(negative_slope=0.1,
                                                            inplace=False),
                                               nn.Conv2d(in_channels=96,
                                                       out_channels=64,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=16,
                                                       padding_mode="zeros",
                                                       dilation=16),
                                               nn.LeakyReLU(negative_slope=0.1,
                                                            inplace=False),
                                               nn.Conv2d(in_channels=64,
                                                       out_channels=32,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1,
                                                       padding_mode="zeros",
                                                       dilation=1),
                                               nn.LeakyReLU(negative_slope=0.1,
                                                            inplace=False),
                                               nn.Conv2d(in_channels=32,
                                                       out_channels=2,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1,
                                                       padding_mode="zeros",
                                                       dilation=1))

    def forward(self,
                f_w_2: torch.Tensor) -> torch.Tensor:

        return self.context_network(f_w_2)

class PWCNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.fpen = FeaturePyramidExtractorNetwork()

        self.ofe2 = OpticalFLowEstimator(in_channels=81+2+32)
        self.ofe3 = OpticalFLowEstimator(in_channels=81+2+64)
        self.ofe4 = OpticalFLowEstimator(in_channels=81+2+96)
        self.ofe5 = OpticalFLowEstimator(in_channels=81+2+128)
        self.ofe6 = OpticalFLowEstimator(in_channels=81+2+192)

        self.cn = ContextNetwork()

    def _create_identity_grid(self,
                          batch: int,
                          height: int,
                          width: int,
                          device: torch.device = "cuda"):
    
        y = torch.linspace(-1, 1, height , device= device)
        x = torch.linspace(-1, 1, width, device = device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        grid = grid.repeat(batch, 1, 1, 1)
        
        return grid.permute(0, 3, 1, 2)

    def warp_image(self,
                   img: torch.Tensor, 
                   flow: torch.Tensor):
        
        batch, _, height, width = img.shape
        
        grid = self._create_identity_grid(batch = batch,
                                          height = height,
                                          width = width,
                                          device = img.device)
        
        flow_permuted = flow
        
        flow_normalized = torch.zeros_like(flow_permuted).to(img.device)
        flow_normalized[:, 0] = 2.0 * flow_permuted[:, 0] / (width - 1)
        flow_normalized[:, 1] = 2.0 * flow_permuted[:, 1] / (height - 1)
        
        sampling_grid = grid + flow_normalized
        
        warped = F.grid_sample(input = img,
                               grid = sampling_grid.permute(0, 2, 3, 1), 
                               align_corners=True, 
                               padding_mode='border')
        return warped

    def cost_volume(self,
                    c1: torch.Tensor, 
                    c2w: torch.Tensor, 
                    search_range: int = 4):
        """
        Args:
            c1: features from image 1, shape [B, C, H, W]
            c2w: warped features from image 2, shape [B, C, H, W]
            search_range: d (maximum displacement in each direction)
        
        Returns:
            cost_vol: shape [B, (2*d+1)^2, H, W]
        """
        B, C, H, W = c1.shape
        d = search_range
        
        # Normalize features
        c1 = c1 / (torch.norm(c1, dim=1, keepdim=True) + 1e-8)
        c2w = c2w / (torch.norm(c2w, dim=1, keepdim=True) + 1e-8)
        
        cost_vol = []
        
        # Search in [-d, d] range
        for dy in range(-d, d+1):
            for dx in range(-d, d+1):
                # Shift c2w by (dx, dy)
                shifted = torch.zeros_like(c2w)
                
                if dy < 0:
                    shifted[:, :, :dy, :] = c2w[:, :, -dy:, :]
                elif dy > 0:
                    shifted[:, :, dy:, :] = c2w[:, :, :-dy, :]
                else:
                    shifted[:, :, :, :] = c2w
                
                if dx < 0:
                    shifted[:, :, :, :dx] = shifted[:, :, :, -dx:]
                elif dx > 0:
                    shifted[:, :, :, dx:] = shifted[:, :, :, :-dx]
                
                # Compute correlation (dot product across channel dimension)
                corr = (c1 * shifted).sum(dim=1, keepdim=True)
                cost_vol.append(corr)
        
        cost_vol = torch.cat(cost_vol, dim=1)  # [B, (2d+1)^2, H, W]
        
        return cost_vol

    def forward(self,
                I_1_2: torch.Tensor) -> torch.Tensor:

        I_1 = I_1_2[:, :3]
        I_2 = I_1_2[:, 3:]

        c_1_6, c_1_5, c_1_4, c_1_3, c_1_2 = self.fpen(I_1)
        c_2_6, c_2_5, c_2_4, c_2_3, c_2_2 = self.fpen(I_2)

        B, _, H_6, W_6 = c_1_6.shape
        
        c_v_2_6 = self.cost_volume(c2w = c_2_6, 
                                   c1 = c_1_6,
                                   search_range= 4)

        init_flow_6 = torch.zeros((B, 2, H_6, W_6),
                                dtype= torch.float32,
                                device= I_1.device)

        flow_input_6 = torch.cat((c_1_6, c_v_2_6, init_flow_6), dim= 1)

        flow_6 = self.ofe6(flow_input_6) + init_flow_6

        init_flow_5 = F.interpolate(input=flow_6,
                               scale_factor= 2,
                               mode= "bilinear",
                               align_corners=True)
                
        warp_5 = self.warp_image(img= c_2_5,
                                 flow= init_flow_5 * 0.625)

        c_v_2_5 = self.cost_volume(c2w= warp_5,
                                   c1= c_1_5,
                                   search_range= 4)

        flow_input_5 = torch.cat((c_1_5, c_v_2_5, init_flow_5), dim= 1)

        flow_5 = self.ofe5(flow_input_5) + init_flow_5

        init_flow_4 = F.interpolate(input=flow_5,
                               scale_factor= 2,
                               mode= "bilinear",
                               align_corners=True)
                
        warp_4 = self.warp_image(img= c_2_4,
                                 flow= init_flow_4 * 1.25)

        c_v_2_4 = self.cost_volume(c2w= warp_4,
                                   c1= c_1_4,
                                   search_range= 4)

        flow_input_4 = torch.cat((c_1_4, c_v_2_4, init_flow_4), dim= 1)

        flow_4 = self.ofe4(flow_input_4) + init_flow_4

        init_flow_3 = F.interpolate(input=flow_4,
                               scale_factor= 2,
                               mode= "bilinear",
                               align_corners=True)
                
        warp_3 = self.warp_image(img= c_2_3,
                                 flow= init_flow_3 * 2.5)

        c_v_2_3 = self.cost_volume(c2w= warp_3,
                                   c1= c_1_3,
                                   search_range= 4)

        flow_input_3 = torch.cat((c_1_3, c_v_2_3, init_flow_3), dim= 1)

        flow_3 = self.ofe3(flow_input_3) + init_flow_3

        init_flow_2 = F.interpolate(input=flow_3,
                               scale_factor= 2,
                               mode= "bilinear",
                               align_corners=True)
                
        warp_2 = self.warp_image(img= c_2_2,
                                 flow= init_flow_2 * 5.0)

        c_v_2_2 = self.cost_volume(c2w= warp_2,
                                   c1= c_1_2,
                                   search_range= 4)

        flow_input_2 = torch.cat((c_1_2, c_v_2_2, init_flow_2), dim= 1)

        flow_2 = self.ofe2(flow_input_2) + init_flow_2

        cn_input = torch.cat((c_1_2, flow_2), dim= 1)

        refined_flow_2 = self.cn(cn_input) + flow_2

        # print(f"flow_6: min={flow_6.min():.3f}, max={flow_6.max():.3f}, mean={flow_6.mean():.3f}, std={flow_6.std():.3f}")
        # print(f"flow_5: min={flow_5.min():.3f}, max={flow_5.max():.3f}, mean={flow_5.mean():.3f}, std={flow_5.std():.3f}")
        # print(f"flow_4: min={flow_4.min():.3f}, max={flow_4.max():.3f}, mean={flow_4.mean():.3f}, std={flow_4.std():.3f}")
        # print(f"flow_3: min={flow_3.min():.3f}, max={flow_3.max():.3f}, mean={flow_3.mean():.3f}, std={flow_3.std():.3f}")
        # print(f"flow_2: min={flow_2.min():.3f}, max={flow_2.max():.3f}, mean={flow_2.mean():.3f}, std={flow_2.std():.3f}")
        # print(f"refined_flow_2: min={refined_flow_2.min():.3f}, max={refined_flow_2.max():.3f}, mean={refined_flow_2.mean():.3f}, std={refined_flow_2.std():.3f}")
        
        return refined_flow_2, flow_3, flow_4, flow_5, flow_6
