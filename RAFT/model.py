import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int, 
                 intermediate_channels: int,
                 downsample: bool = False):

        super().__init__()
        self.downsample = downsample
        if self.downsample:
            stride = 2
        else:
            stride = 1
        
        self.resblock = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                out_channels=intermediate_channels,
                                                kernel_size=3,
                                                stride=stride,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm2d(num_features=intermediate_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=intermediate_channels,
                                                out_channels=intermediate_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.BatchNorm2d(num_features=intermediate_channels),
                                      nn.ReLU(inplace=True))

        self.relu = nn.ReLU(inplace=True)

        self.downsampler = nn.Conv2d(in_channels=in_channels,
                                    out_channels=intermediate_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    padding_mode="zeros")

    def forward(self,
                x_in: torch.Tensor) -> torch.Tensor:

        x = self.resblock(x_in)
        
        if self.downsample:
            x_in = self.downsampler(x_in)

        return self.relu(x + x_in)
                                      
class FeatureExtractor(nn.Module):

    def __init__(self,
                 in_channels: int):

        super().__init__()

        self.shallow_features = nn.Conv2d(in_channels=in_channels,
                                          out_channels=64,
                                          kernel_size=7,
                                          stride=2,
                                          padding=3,
                                          padding_mode="zeros")
        
        self.resblock_1 = nn.Sequential(ResidualBlock(in_channels=64,
                                                      intermediate_channels=64,
                                                      downsample=False),
                                        ResidualBlock(in_channels=64,
                                                      intermediate_channels=64,
                                                      downsample=False))

        self.resblock_2 = nn.Sequential(ResidualBlock(in_channels=64,
                                                      intermediate_channels=96,
                                                      downsample=True),
                                        ResidualBlock(in_channels=96,
                                                      intermediate_channels=96,
                                                      downsample=False))

        self.resblock_3 = nn.Sequential(ResidualBlock(in_channels=96,
                                                      intermediate_channels=128,
                                                      downsample=True),
                                        ResidualBlock(in_channels=128,
                                                      intermediate_channels=128,
                                                      downsample=False))

        self.last_conv = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   padding_mode="zeros")

    def forward(self,
                x_in: torch.Tensor) -> torch.Tensor:

        x = self.resblock_1(self.shallow_features(x_in))
        x = self.last_conv(self.resblock_3(self.resblock_2(x)))

        return x

class ConvGRU(nn.Module):

    def __init__(self,
                 in_channels: int,
                 kernel_size: Tuple[int, int]):

        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=128,
                                kernel_size=kernel_size,
                                stride=1,
                                padding="same",
                                padding_mode="zeros")

        self.conv_2 = nn.Conv2d(in_channels=in_channels,
                                out_channels=128,
                                kernel_size=kernel_size,
                                stride=1,
                                padding="same",
                                padding_mode="zeros")

        self.conv_3 = nn.Conv2d(in_channels=in_channels,
                                out_channels=128,
                                kernel_size=kernel_size,
                                stride=1,
                                padding="same",
                                padding_mode="zeros")

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                h_t_1: torch.Tensor,
                x_t: torch.Tensor) -> torch.Tensor:
        
        input_cat = torch.cat((h_t_1, x_t), dim=1)
        
        z_t = self.sigmoid(self.conv_1(input_cat))
        r_t = self.sigmoid(self.conv_2(input_cat))
        h_t_tilda = self.tanh(self.conv_3(torch.cat((r_t * h_t_1, x_t), dim=1)))
        h_t = (1 - z_t) * h_t_1 + z_t * h_t_tilda

        # h_t = torch.tanh(h_t)

        return h_t

class ConvGRUPair(nn.Module):

    def __init__(self,
                 in_channels: int,
                 kernel_size_1: Tuple[int, int],
                 kernel_size_2: Tuple[int, int]):

        super().__init__()

        self.convgru_1 = ConvGRU(in_channels=in_channels,
                                 kernel_size=kernel_size_1)

        self.convgru_2 = ConvGRU(in_channels=in_channels,
                                 kernel_size=kernel_size_2)

    def forward(self,
                h_t_1: torch.Tensor,
                x_t: torch.Tensor) -> torch.Tensor:

        h_t_intermediate = self.convgru_1(h_t_1, x_t)
        h_t = self.convgru_2(h_t_intermediate, x_t)

        return h_t

class CorrelationBlock:
    def __init__(self,
                 fmap_1: torch.Tensor,
                 fmap_2: torch.Tensor,
                 num_levels: int,
                 radius: int):

        self.num_levels = num_levels
        self.radius = radius

        B, C, H, W = fmap_1.shape

        corr = torch.einsum('bchw, bcHW -> bhwHW', fmap_1, fmap_2)
        corr = corr / (C ** 0.5)
        corr = corr.reshape(B * H * W, 1, H, W)

        self.corr_pyramids = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(input= corr,
                                kernel_size=2,
                                stride=2)

            self.corr_pyramids.append(corr)

    def __warp__(self,
                 img: torch.Tensor,
                 coords: torch.Tensor,
                 mode: str = "bilinear",
                 mask: bool = False):

        H, W = img.shape[-2:]
        x_grid, y_grid = coords.split([1, 1], dim= -1)

        x_grid, y_grid = 2*x_grid/(W-1)-1, 2*y_grid/(H-1)-1

        grid = torch.cat([x_grid, y_grid], dim=-1)

        img = F.grid_sample(input=img,
                            grid=grid,
                            align_corners=True,
                            mode=mode)

        if mask:
            mask = (x_grid > -1) & (y_grid > -1) & (x_grid < 1) & (y_grid < 1)
            return img, mask.float()

        return img
        
    def __call__(self,
                 coords: torch.Tensor):

        r = self.radius
        B, _, H, W = coords.shape
        coords = coords.permute(0, 2, 3, 1)

        out_pyramid = []

        for level, corr in enumerate(self.corr_pyramids):

            dy = torch.arange(-r, r+1, device=coords.device)
            dx = torch.arange(-r, r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), dim=-1)

            coords_scaled = coords / (2 ** level)

            coords_scaled = coords_scaled.reshape(B * H * W, 1, 1, 2)
            delta = delta.reshape(1, 2*r+1, 2*r+1, 2)
            # print(delta.shape, coords_scaled.shape)
            sample_coords = coords_scaled + delta
            # print(sample_coords.shape, corr.shape)
            corr_sampled = self.__warp__(img=corr,
                                         coords=sample_coords,
                                         mode="bilinear")
            # print(corr_sampled.shape)
            corr_sampled = corr_sampled.reshape(B, H, W, -1)

            out_pyramid.append(corr_sampled)

        out = torch.cat(out_pyramid, dim=-1)
        
        return out.permute(0, 3, 1, 2)

class BasicMotionEncoder(nn.Module):

    def __init__(self,
                 in_channels: int):

        super().__init__()

        self.conv_corr = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=256,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding="same",
                                                 padding_mode="zeros"),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=256,
                                                 out_channels=128,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding="same",
                                                 padding_mode="zeros"),
                                       nn.ReLU(inplace=True))

        self.conv_flow = nn.Sequential(nn.Conv2d(in_channels=2,
                                                 out_channels=128,
                                                 kernel_size=7,
                                                 stride=1,
                                                 padding="same",
                                                 padding_mode="zeros"),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=128,
                                                 out_channels=64,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding="same",
                                                 padding_mode="zeros"),
                                       nn.ReLU(inplace=True))

        self.conv_corr_flow = nn.Sequential(nn.Conv2d(in_channels=128 + 64,
                                                      out_channels=128-2,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding="same",
                                                      padding_mode="zeros"),
                                            nn.ReLU(inplace=True))

    def forward(self,
               corr: torch.Tensor,
               flow: torch.Tensor) -> torch.Tensor:

        corr_feat = self.conv_corr(corr)
        flow_feat = self.conv_flow(flow)
        corr_flow_feat = torch.cat([corr_feat, flow_feat], dim=1)
        corr_flow_feat_processed = self.conv_corr_flow(corr_flow_feat)

        out = torch.cat([corr_flow_feat_processed, flow], dim=1)
        # print(out.shape)
        return out

class UpdateBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.motion_encoder = BasicMotionEncoder(in_channels=4 * 81)
        self.gru = ConvGRUPair(in_channels=384,
                               kernel_size_1=(1, 5),
                               kernel_size_2=(5, 1))

        self.flow_head = nn.Sequential(nn.Conv2d(in_channels=128,
                                                 out_channels=256,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding="same",
                                                 padding_mode="zeros"),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=256,
                                                 out_channels=2,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding="same",
                                                 padding_mode="zeros"))

        self.upsample_block = nn.Sequential(nn.Conv2d(in_channels=128,
                                            out_channels=256,
                                            kernel_size=3,
                                            stride=1,
                                            padding="same",
                                            padding_mode="zeros"),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=256,
                                            out_channels=64 * 9,
                                            kernel_size=1,
                                            stride=1,
                                            padding="same",
                                            padding_mode="zeros"))

    def forward(self,
                hidden_state: torch.Tensor,
                con_net_in: torch.Tensor,
                delta_flow: torch.Tensor,
                corr: torch.Tensor):

        motion_feat = self.motion_encoder(corr, delta_flow)
        # print(motion_feat.shape, con_net_in.shape)
        gru_in = torch.cat([con_net_in, motion_feat], dim=1)

        hidden_state_new = self.gru(hidden_state, gru_in)
        delta_flow_new = self.flow_head(hidden_state_new)

        upsampled_flow_mask = 0.25 * self.upsample_block(hidden_state_new)

        return hidden_state_new, delta_flow_new, upsampled_flow_mask

class RAFT(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 contest_dim: int,
                 iters: int):

        super().__init__()
        self.hidden_dim, self.contest_dim = hidden_dim, contest_dim
        self.iters = iters
        
        self.feature_extractor = FeatureExtractor(in_channels=3)
        self.context_net = FeatureExtractor(in_channels=3)
        self.update_block = UpdateBlock()

    def _init_flow(self,
                   img: torch.Tensor):

        B, C, H, W = img.shape

        init_grid = torch.meshgrid([torch.arange(H//8, device= img.device), torch.arange(W//8, device= img.device)], indexing='ij')
        init_grid = torch.stack(init_grid[::-1], dim=0).float() 
        
        coord0 = init_grid.repeat(B, 1, 1, 1)
        coord1 = init_grid.repeat(B, 1, 1, 1)

        return coord0, coord1 

    def _upsample_flow(self,
                       flow: torch.Tensor,
                       flow_mask: torch.Tensor):

        B, _, H, W = flow.shape

        flow_mask = flow_mask.view(B, 1, 9, 8, 8, H, W)
        flow_mask = flow_mask.softmax(dim=2)
        # print(flow.shape)
        up_flow = F.unfold(8 * flow, kernel_size=(3, 3), padding=1)
        # print(up_flow.shape)
        up_flow = up_flow.view(B, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(flow_mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_flow = up_flow.reshape(B, 2, 8 * H, 8 * W)

        return up_flow

    def forward(self,
                both_img: torch.Tensor):

        img_1, img_2 = both_img.chunk(2, dim=1)
        
        fmap_1, fmap_2 = self.feature_extractor(img_1), self.feature_extractor(img_2)

        corr_fn = CorrelationBlock(fmap_1=fmap_1, fmap_2=fmap_2, num_levels=4, radius=4)

        context = self.context_net(img_1)
        
        context_net_in, hidden_state = torch.split(tensor=context, 
                                                   split_size_or_sections=[self.contest_dim, self.hidden_dim],
                                                   dim=1)

        context_net_in, hidden_state = F.relu(context_net_in), F.tanh(hidden_state)
        
        coords0, coords1 = self._init_flow(img= img_2)

        flow_predictions = []
        for iter in range(self.iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)

            flow = coords1 - coords0

            hidden_state, delta_flow, upsampled_flow_mask = self.update_block(hidden_state, context_net_in, flow, corr)

            # delta_flow = torch.clamp(delta_flow, min=-100, max=100)

            coords1 = coords1 + delta_flow

            flow_up = self._upsample_flow(coords1 - coords0, upsampled_flow_mask)
            # print(flow_up.shape)
            flow_predictions.append(flow_up)

        return flow_predictions
