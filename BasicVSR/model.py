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


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        
        if C == 8:
            return self.conv_net(x)
        else:
            flow = self._create_identity_grid(batch= B,
                                              height= H,
                                              width= W,
                                              device= x.device)
            
            input_cat = torch.cat((x, flow), dim= 1)
            return self.conv_net(input_cat)

class SPyNet(nn.Module):

    def __init__(self,
                 layers: int):

        super().__init__()
        self.layers = layers

        self.spy_net = nn.ModuleList([ConvNet() for _ in range(layers)])

        self.tanh = nn.Tanh()

    def _create_identity_grid(self,
                          batch: int,
                          height: int,
                          width: int,
                          device: torch.device):
    
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
        
        sampling_grid = grid + flow
        
        warped = F.grid_sample(input = img,
                               grid = sampling_grid.permute(0, 2, 3, 1), 
                               align_corners=True, 
                               padding_mode='border')
        return warped

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:

        total_flow = 0.0
        B, _, H, W = X.shape
        flow = None
        for i, layer_num in enumerate(range(self.layers, 0, -1)):
            if layer_num == self.layers:
                flow = flow = torch.zeros(B, 2, H // (2 ** (layer_num - 1)), W // (2 ** (layer_num - 1)),
                                          device=X.device)
            else:
                flow = F.interpolate(input=flow,
                                     scale_factor= 2,
                                     mode= "bilinear",
                                     align_corners = True) * 2.0
                
            X_downsampled = F.interpolate(input=X,
                                          size= ((H // (2 ** (layer_num -1))), (W // (2 ** (layer_num -1)))),
                                          mode= "bilinear")
            
            # print(upsampled_flow.shape)
            warped_image = self.warp_image(img= X_downsampled[:, 3:6], 
                                           flow= flow)
            if layer_num == self.layers:
                concat_input = torch.cat((X_downsampled, flow), dim= 1)

            else:
                concat_input = torch.cat((X_downsampled[:, 3:6], warped_image, flow), dim= 1)
            
            residual_flow = self.spy_net[i](concat_input)
            flow = residual_flow + flow

            flow = self.tanh(flow)
            
        return flow

class FNet(nn.Module):

    def __init__(self,
                in_channels: int):

        super().__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                              out_channels= 32,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False),
                                    nn.Conv2d(in_channels=32,
                                              out_channels= 32,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False)
                                   )

        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=32,
                                              out_channels= 64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False),
                                    nn.Conv2d(in_channels=64,
                                              out_channels= 64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False)
                                   )

        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels= 128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False),
                                    nn.Conv2d(in_channels=128,
                                              out_channels= 128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False)
                                   )

        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels= 256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False),
                                    nn.Conv2d(in_channels=256,
                                              out_channels= 256,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False)
                                   )

        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels=256,
                                              out_channels=128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False),
                                    nn.Conv2d(in_channels=128,
                                              out_channels= 128,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False)
                                   )

        self.conv_6 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels=64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False),
                                    nn.Conv2d(in_channels=64,
                                              out_channels= 64,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False)
                                   )

        self.classifier = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels=32,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros"),
                                    nn.LeakyReLU(negative_slope=0.2,
                                                 inplace=False),
                                    nn.Conv2d(in_channels=32,
                                              out_channels= 2,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              padding_mode="zeros")
                                   )

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode="bilinear",
                                    align_corners=True)

        self.tanh = nn.Tanh()

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = self.maxpool(self.conv_1(x))
        x = self.maxpool(self.conv_2(x))
        x = self.maxpool(self.conv_3(x))
        x = self.upsample(self.conv_4(x))
        x = self.upsample(self.conv_5(x))
        x = self.upsample(self.conv_6(x))
        x = self.tanh(self.classifier(x))

        return x

class ResidualBlock(nn.Module):

    def __init__(self,
                channels: int):

        super().__init__()
        self.channels = channels
        self.residual_block_1 = nn.Sequential(nn.Conv2d(in_channels=channels + 3,
                                                      out_channels= channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      padding_mode="zeros"),
                                            nn.ReLU(inplace=False),
                                            nn.Conv2d(in_channels=channels,
                                                      out_channels=channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      padding_mode="zeros"))

        self.residual_block_2 = nn.Sequential(nn.Conv2d(in_channels=channels,
                                                      out_channels= channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      padding_mode="zeros"),
                                            nn.ReLU(inplace=False),
                                            nn.Conv2d(in_channels=channels,
                                                      out_channels=channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      padding_mode="zeros"))

        self.residual_block_3 = nn.Sequential(nn.Conv2d(in_channels=channels,
                                                      out_channels= channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      padding_mode="zeros"),
                                            nn.ReLU(inplace=False),
                                            nn.Conv2d(in_channels=channels,
                                                      out_channels=channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      padding_mode="zeros"))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.residual_block_3(self.residual_block_2(self.residual_block_1(x))) + x[:, :self.channels]

class Upsample(nn.Module):

    def __init__(self,
                 in_channels: int):

        super().__init__()

        self.upsample = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.PixelShuffle(upscale_factor=2),
                                      nn.Conv2d(in_channels=16,
                                                out_channels=12,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                padding_mode="zeros"),
                                      nn.PixelShuffle(upscale_factor=2))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.upsample(x)

class BasicVSR(nn.Module):

    def __init__(self,
                 seq_length: int,
                 flow_estimator: str = "spynet",
                 coupled_propogation: bool = False):

        super().__init__()
        self.seq_length = seq_length
        self.coupled_propogation = coupled_propogation
        
        if flow_estimator == "fnet":
            self.flow_estimator = FNet(in_channels=6)
        elif flow_estimator == "spynet":
            self.flow_estimator = SPyNet(layers=6)
        else:
            print(f"flow estimator can be either 'fnet' or 'spynet'. Defaulting to 'spynet'.")
            self.flow_estimator = SPyNet(layers=6)
        

        self.forward_branch = nn.ModuleList([ResidualBlock(channels=64) for _ in range(seq_length)])

        self.backward_branch = nn.ModuleList([ResidualBlock(channels=64) for _ in range(seq_length)])

        self.upsample_branch = nn.ModuleList([Upsample(in_channels=64 * 2) for _ in range(seq_length)])

    def _create_identity_grid(self,
                              batch: int,
                              height: int,
                              width: int,
                              device: torch.device):
    
        y = torch.linspace(-1, 1, height , device = device)
        x = torch.linspace(-1, 1, width, device = device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        grid = grid.repeat(batch, 1, 1, 1)
        
        return grid.permute(0, 3, 1, 2)

    def warp_features(self,
                      features: torch.Tensor, 
                      flow: torch.Tensor):
        
        batch, _, height, width = features.shape
        
        grid = self._create_identity_grid(batch = batch,
                                          height = height,
                                          width = width,
                                          device = features.device)
        
        sampling_grid = grid + flow
        
        warped = F.grid_sample(input = features,
                               grid = sampling_grid.permute(0, 2, 3, 1), 
                               align_corners=True, 
                               padding_mode='border')
        return warped

    def forward(self,
                x_t: torch.Tensor) -> torch.Tensor:

        B, T, C, H, W = x_t.shape
        h_f_i_list = []
        h_b_i_list = []
        x_tilda_i_list = []
        
        for f_idx, b_idx in enumerate(range(self.seq_length-1, -1, -1)):
            #print(f_idx, b_idx)
            if f_idx == 0 and b_idx == (self.seq_length - 1):
                x_t_minus_1 = torch.zeros_like(x_t[:, f_idx], device= x_t.device)
                x_t_plus_1 = torch.zeros_like(x_t[:, b_idx], device= x_t.device)

                h_f_i_minus_1 = torch.zeros((B, 64, H, W), device= x_t.device)
                h_b_i_plus_1 = torch.zeros((B, 64, H, W), device= x_t.device)

            f_spynet_input = torch.cat([x_t[:, f_idx], x_t_minus_1], dim=1)
            b_spynet_input = torch.cat([x_t[:, b_idx], x_t_plus_1], dim=1)

            s_f_i = self.flow_estimator(f_spynet_input)
            s_b_i = self.flow_estimator(b_spynet_input)

            h_bar_f_i = self.warp_features(features=h_f_i_minus_1,
                                           flow= s_f_i)

            h_bar_b_i = self.warp_features(features=h_b_i_plus_1,
                                           flow=s_b_i)
            #print(h_bar_b_i.shape, h_bar_f_i.shape)

            R_f_input = torch.cat([h_bar_f_i, x_t[:, f_idx]], dim=1)
            R_b_input = torch.cat([h_bar_b_i, x_t[:, b_idx]], dim=1)

            h_f_i = self.forward_branch[f_idx](R_f_input)
            h_b_i = self.backward_branch[b_idx](R_b_input)

            h_f_i_list.append(h_f_i)
            h_b_i_list.append(h_b_i)

            h_f_i_minus_1 = h_f_i
            h_b_i_plus_1 = h_b_i

            x_t_minus_1 = x_t[:, f_idx]
            x_t_plus_1 = x_t[:, b_idx]

        for f_idx, b_idx in enumerate(range(self.seq_length-1, -1, -1)):

            upsample_input = torch.cat([h_f_i_list[f_idx], h_b_i_list[b_idx]], dim=1)

            x_tilda_i = self.upsample_branch[f_idx](upsample_input)

            x_tilda_i_list.append(x_tilda_i)

        output_sequence = torch.stack(x_tilda_i_list, dim=1)

        return output_sequence
