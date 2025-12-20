import torch
import torch.nn as nn

class ShallowFeatureExtractor(nn.Module):
    
    def __init__(self):
        
        super().__init__()

        self.feature_extractor = nn.Conv2d(in_channels=3,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            padding_mode="zeros")

    def forward(self,
                I_s_r: torch.Tensor) -> torch.Tensor:

        F_0 = self.feature_extractor(I_s_r)
        # print(F_0.shape)

        return(F_0)

class ChannelAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(in_channels=64,
                                                         out_channels=4,
                                                         kernel_size=1,
                                                         stride=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(in_channels=4,
                                                         out_channels=64,
                                                         kernel_size=1,
                                                         stride=1),
                                               nn.Sigmoid()
                                              )

    def forward(self,
                x_c: torch.Tensor) -> torch.Tensor:

        s_x_c = torch.mul(self.channel_attention(x_c), x_c)
        # print(s_x_c.shape)

        return s_x_c

class ResidualChannelAttentionBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.stacked_conv = nn.Sequential(nn.Conv2d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    padding_mode="zeros"),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    padding_mode="zeros")
                                         )

        self.channel_attention = ChannelAttention()

    def forward(self,
                F_g_b_minus_1: torch.Tensor) -> torch.Tensor:

        X_g_b = self.stacked_conv(F_g_b_minus_1)
        F_g_b = F_g_b_minus_1 + self.channel_attention(X_g_b)
        # print(F_g_b.shape)

        return F_g_b

class ResidualGroup(nn.Module):

    def __init__(self,
                num_rcab: int):

        super().__init__()

        self.residual_group = nn.Sequential(*[ResidualChannelAttentionBlock() for rcab in range(num_rcab)])

        self.final_conv = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode="zeros")

    def forward(self,
                F_g_minus_1: torch.Tensor) -> torch.Tensor:

        F_g = self.residual_group(F_g_minus_1)
        rg_out = self.final_conv(F_g) + F_g_minus_1
        # print(rg_out.shape)
        
        return rg_out

class ResidualInResidual(nn.Module):

    def __init__(self,
                 num_rg: int,
                 num_rcab: int):

        super().__init__()

        self.rir = nn.Sequential(*[ResidualGroup(num_rcab= num_rcab) for rg in range(num_rg)])

        self.final_conv = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode="zeros")

    def forward(self,
                F_0: torch.Tensor) -> torch.Tensor:
        
        F_d_f = F_0 + self.final_conv(self.rir(F_0))
        # print(F_d_f.shape)

        return F_d_f


class UpscaleAndReconstruct(nn.Module):

    def __init__(self):

        super().__init__()

        self.upsampler = nn.Sequential(nn.Conv2d(in_channels=64,
                                                 out_channels=64 * 2 * 2,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 padding_mode="zeros"),
                                       nn.PixelShuffle(upscale_factor=2)
                                      )

        self.recontruct = nn.Conv2d(in_channels=64,
                                    out_channels=3,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode="zeros")

    def forward(self,
                F_d_f: torch.Tensor) -> torch.Tensor:

        I_s_r = self.recontruct(self.upsampler(F_d_f))
        # print(I_s_r.shape)
        
        return I_s_r

class ResidualChannelAttentionNetwork(nn.Module):

    def __init__(self,
                 num_rg: int,
                 num_rcab: int):

        super().__init__()

        self.shallow_feature_extractor = ShallowFeatureExtractor()

        self.residual_in_residual = ResidualInResidual(num_rg= num_rg,
                                                       num_rcab= num_rcab)

        self.upsample_and_reconstruct = UpscaleAndReconstruct()

    def forward(self,
                I_l_r: torch.Tensor) -> torch.Tensor:

        I_s_r = self.upsample_and_reconstruct(self.residual_in_residual(self.shallow_feature_extractor(I_l_r)))

        return I_s_r
                                                        
