import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_net = nn.Sequential(nn.Conv2d(in_channels=8,
                                               out_channels=32,
                                               kernel_size= 7,
                                               stride=1,
                                               padding = 3,
                                               padding_mode= "zeros"),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32,
                                               out_channels=64,
                                               kernel_size= 7,
                                               stride=1,
                                               padding = 3,
                                               padding_mode= "zeros"),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64,
                                               out_channels=32,
                                               kernel_size= 7,
                                               stride=1,
                                               padding = 3,
                                               padding_mode= "zeros"),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=32,
                                               out_channels=16,
                                               kernel_size= 7,
                                               stride=1,
                                               padding = 3,
                                               padding_mode= "zeros"),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=16,
                                               out_channels=2,
                                               kernel_size= 7,
                                               stride=1,
                                               padding = 3,
                                               padding_mode= "zeros")
                                    )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.conv_net(x)

class SPyNet(nn.Module):

    def __init__(self,
                 layers: int):

        super().__init__()
        self.layers = layers

        self.spy_net = nn.ModuleList([ConvNet() for _ in range(layers)])

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
        
        flow_permuted = -flow
        
        flow_normalized = torch.zeros_like(flow_permuted).to(img.device)
        flow_normalized[:, 0] = 2.0 * flow_permuted[:, 0] / (width - 1)
        flow_normalized[:, 1] = 2.0 * flow_permuted[:, 1] / (height - 1)
        
        sampling_grid = grid + flow_normalized
        
        warped = F.grid_sample(input = img,
                               grid = sampling_grid.permute(0, 2, 3, 1), 
                               align_corners=False, 
                               padding_mode='border')
        return warped

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:

        total_flow = 0.0
        B, _, H, W = X.shape
        flow = None
        for i, layer_num in enumerate(range(self.layers, 0, -1)):
            if layer_num == self.layers:
                flow = self._create_identity_grid(batch= B,
                                                  height= H // (2 ** (layer_num)),
                                                  width= W // (2 ** (layer_num)))
            X_downsampled = F.interpolate(input=X,
                                          size= ((H // (2 ** (layer_num -1))), (W // (2 ** (layer_num -1)))),
                                          mode= "bilinear")
            upsampled_flow = F.interpolate(input=flow,
                                           scale_factor= 2,
                                           mode= "bilinear")
            # print(upsampled_flow.shape)
            warped_image = self.warp_image(img= X_downsampled[:, :3], 
                                           flow= upsampled_flow)
            if layer_num == self.layers:
                concat_input = torch.cat((X_downsampled, upsampled_flow), dim= 1)

            else:
                concat_input = torch.cat((X_downsampled[:, :3], warped_image, upsampled_flow), dim= 1)
            
            residual_flow = self.spy_net[i](concat_input)
            flow = residual_flow + upsampled_flow
            
        return flow
