import torch
import torch.nn as nn
import torch.nn.functional as F

def init_params(model: torch.nn.Module):

    for module in model.modules():
        if isinstance(model, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

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

        self.residual_block = nn.Sequential(nn.Conv2d(in_channels=channels,
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

        return self.residual_block(x) + x

class SRNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 res_in_channels: int,
                 residual_blocks: int):

        super().__init__()

        self.shallow_feature_extractor = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                                 out_channels= res_in_channels,
                                                                 kernel_size=3,
                                                                 stride=1,
                                                                 padding=1,
                                                                 padding_mode="zeros"),
                                                        nn.LeakyReLU(negative_slope=0.2,
                                                                     inplace=False))

        self.all_blocks = nn.ModuleList([ResidualBlock(channels=res_in_channels) for block in range(residual_blocks)])

        self.upsample = nn.Sequential(nn.ConvTranspose2d(in_channels=res_in_channels,
                                                       out_channels=64,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1,
                                                       padding_mode="zeros"),
                                    nn.ReLU(inplace=False),
                                    nn.ConvTranspose2d(in_channels=64,
                                                       out_channels=64,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1,
                                                       padding_mode="zeros"),
                                     nn.ReLU(inplace=False),
                                     nn.Conv2d(in_channels=64,
                                               out_channels=3,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               padding_mode="zeros"))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = self.shallow_feature_extractor(x)
        for block in self.all_blocks:
            x = block(x)

        x = self.upsample(x)
        
        return x

class FRVSR(nn.Module):

    def __init__(self,
                 residual_blocks: int, 
                 res_in_channels: int,
                 scale_factor: int):

        super().__init__()
        self.scale_factor = scale_factor
        
        self.fnet = FNet(in_channels=6)

        self.srnet = SRNet(in_channels=3 + 3 * scale_factor ** 2,
                           res_in_channels=res_in_channels,
                           residual_blocks=residual_blocks)

        self.upsample = nn.Upsample(scale_factor=scale_factor,
                                    mode="bilinear",
                                    align_corners=True)

        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=scale_factor)

    def _create_identity_grid(self,
                              batch: int,
                              height: int,
                              width: int,
                              device: torch.device):

        y = torch.linspace(-1, 1, height, device = device)
        x = torch.linspace(-1, 1, width, device = device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        grid = grid.repeat(batch, 1, 1, 1)

        return grid.permute(0, 3, 1, 2)

    def warp(self,
             img: torch.Tensor,
             flow: torch.Tensor) -> torch.Tensor:

        B, _, H, W = img.shape

        grid = self._create_identity_grid(batch= B,
                                          height=H,
                                          width=W,
                                          device=img.device)

        sampling_grid = grid + flow

        warped = F.grid_sample(input= img,
                               grid= sampling_grid.permute(0, 2, 3, 1),
                               align_corners=True,
                               padding_mode='border')

        return warped

    def forward(self,
                in_x: torch.Tensor) -> torch.Tensor:

        B, _, _, H, W = in_x.shape
        I_lr_est_t_list = []
        I_est_t_list = []
        
        for i in range(in_x.shape[1]):
            
            I_lr_t = in_x[:, i]
            
            if i == 0:
                I_lr_t_1 = torch.zeros_like(I_lr_t, device=in_x.device)
                I_est_t_1 = torch.zeros((B, 3, H * self.scale_factor, W * self.scale_factor), device=in_x.device)
            else:
                I_lr_t_1 = in_x[:, i-1]
        
            fnet_input = torch.cat((I_lr_t_1, I_lr_t), dim=1)
          
            F_lr = self.fnet(fnet_input)

            I_lr_est_t = self.warp(img=I_lr_t_1,
                                   flow= F_lr)

            F_hr = self.upsample(F_lr) * self.scale_factor

            I_tilda_est_t_1 = self.warp(img=I_est_t_1,
                                        flow= F_hr)

            S_s = self.space_to_depth(I_tilda_est_t_1)

            srnet_input = torch.cat((I_lr_t, S_s), dim=1)

            I_est_t = self.srnet(srnet_input)

            I_lr_est_t_list.append(I_lr_est_t)
            I_est_t_list.append(I_est_t)

            I_est_t_1 = I_est_t

        I_lr_est_t_tensor = torch.stack(I_lr_est_t_list, dim=1)
        I_est_t_tensor = torch.stack(I_est_t_list, dim=1)

        return I_lr_est_t_tensor, I_est_t_tensor
