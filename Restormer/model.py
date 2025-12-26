import torch
from torch import nn, einsum
from einops import rearrange
from typing import Tuple

class GatedDConvFeedForwardNetwork(nn.Module):

    def __init__(self,
                 in_dimensions: int,
                 gamma: float):

        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape= in_dimensions,
                                       bias= False)

        self.up_pconv = nn.Conv2d(in_channels= in_dimensions,
                               out_channels= int(in_dimensions * gamma) * 2,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias= False)
        
        self.dconv = nn.Conv2d(in_channels= int(in_dimensions * gamma) * 2,
                               out_channels= int(in_dimensions * gamma) * 2,
                               kernel_size= 3,
                               stride=1,
                               padding=1,
                               padding_mode="zeros",
                               groups= int(in_dimensions * gamma) * 2,
                               bias = False)

        self.gelu = nn.GELU()

        self.down_pconv = nn.Conv2d(in_channels= int(in_dimensions * gamma),
                                    out_channels= in_dimensions,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0, 
                                    bias = False)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x_in = x
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        x_1, x_2 = self.dconv(self.up_pconv(x)).chunk(2, dim=1)
        x_2 = self.gelu(x_2)
        x_mul = torch.mul(input= x_1,
                          other= x_2)
        x = self.down_pconv(x_mul)
        x_tilda = torch.add(input=x,
                            other=x_in)
        
        return x_tilda

class MultiDConvHeadTransposedAttention(nn.Module):

    def __init__(self,
                 in_dimensions: int,
                 heads: int):

        super().__init__()
        self.heads = heads
        self.heads_dim = in_dimensions // heads

        self.layer_norm = nn.LayerNorm(normalized_shape= in_dimensions,
                                       bias= False)

        self.pconv = nn.Conv2d(in_channels= in_dimensions,
                              out_channels= in_dimensions * 3,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias = False)

        self.dconv = nn.Conv2d(in_channels= in_dimensions * 3,
                               out_channels= in_dimensions * 3,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode="zeros",
                               groups=in_dimensions * 3,
                               bias= False)

        self.pconv_out = nn.Conv2d(in_channels= in_dimensions,
                                  out_channels= in_dimensions,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias = False)

        self.softmax = nn.Softmax(dim= -1)

        self.alpha = nn.Parameter(torch.randn(1, dtype=torch.float32),
                                 requires_grad=True)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        
        x_in = x
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)

        qkv = self.dconv(self.pconv(x)).permute(0, 2, 3, 1).chunk(3, dim= -1)

        q, k, v = map(lambda t: rearrange(tensor= t, 
                                          pattern= 'B H W (h d) -> B h (H W) d',
                                          h = self.heads, d = self.heads_dim),
                      qkv)

        q_k_dot_product = einsum('B h N i, B h N j -> B h i j',
                                 q, k) / self.alpha

        score = self.softmax(q_k_dot_product)

        attn = einsum('B h N i, B h i j -> B h N j', v, score)

        attn = rearrange(tensor= attn, 
                         pattern= 'B h (H W) j -> B (h j) H W',
                         h = self.heads, j = self.heads_dim, H = H, W = W)

        conv_out = self.pconv_out(attn)

        x_tilda = conv_out + x_in

        return x_tilda

class TransformerLayer(nn.Module):

    def __init__(self,
                 in_dimensions: int,
                 gamma: float,
                 heads: int):

        super().__init__()

        self.mdta = MultiDConvHeadTransposedAttention(in_dimensions= in_dimensions,
                                                      heads= heads)

        self.gdfn = GatedDConvFeedForwardNetwork(in_dimensions= in_dimensions,
                                                 gamma= gamma)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.gdfn(self.mdta(x))

class TransformerBlock(nn.Module):

    def __init__(self,
                 in_dimensions: int,
                 gamma: float,
                 heads: int,
                 layers: int):

        super().__init__()

        self.transformer_block = nn.Sequential(*[TransformerLayer(in_dimensions=in_dimensions,
                                                                  gamma= gamma,
                                                                  heads= heads) for _ in range(layers)])

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.transformer_block(x)

class Restormer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_dimensions: Tuple[int, int, int, int],
                 heads: Tuple[int, int, int, int],
                 t_layers: Tuple[int, int, int, int],
                 gamma: float):

        super().__init__()

        self.shallow_features = nn.Conv2d(in_channels=in_channels,
                                          out_channels= in_dimensions[0],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          padding_mode="zeros",
                                          bias= False)

        self.downsample_1 = nn.Sequential(nn.Conv2d(in_channels= in_dimensions[0],
                                                   out_channels= in_dimensions[0] // 2,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros",
                                                   bias=False),
                                         nn.PixelUnshuffle(downscale_factor= 2))
        self.downsample_2 = nn.Sequential(nn.Conv2d(in_channels= in_dimensions[1],
                                                   out_channels= in_dimensions[1] // 2,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros",
                                                   bias=False),
                                         nn.PixelUnshuffle(downscale_factor= 2))
        self.downsample_3 = nn.Sequential(nn.Conv2d(in_channels= in_dimensions[2],
                                                   out_channels= in_dimensions[2] // 2,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros",
                                                   bias=False),
                                         nn.PixelUnshuffle(downscale_factor= 2))

        self.transformer_block_down_1 = TransformerBlock(in_dimensions= in_dimensions[0],
                                                        layers= t_layers[0],
                                                        heads=heads[0],
                                                        gamma = gamma)
        self.transformer_block_down_2 = TransformerBlock(in_dimensions= in_dimensions[1],
                                                        layers= t_layers[1],
                                                        heads=heads[1],
                                                        gamma = gamma)
        self.transformer_block_down_3 = TransformerBlock(in_dimensions= in_dimensions[2],
                                                        layers= t_layers[2],
                                                        heads=heads[2],
                                                        gamma = gamma)
        self.transformer_block_4 = TransformerBlock(in_dimensions= in_dimensions[3],
                                                        layers= t_layers[3],
                                                        heads=heads[3],
                                                        gamma = gamma)

        self.transformer_block_up_1 = TransformerBlock(in_dimensions= in_dimensions[1],
                                                        layers= t_layers[1],
                                                        heads=heads[1],
                                                        gamma = gamma)
        self.transformer_block_up_2 = TransformerBlock(in_dimensions= in_dimensions[1],
                                                        layers= t_layers[1],
                                                        heads=heads[1],
                                                        gamma = gamma)
        self.transformer_block_up_3 = TransformerBlock(in_dimensions= in_dimensions[2],
                                                        layers= t_layers[2],
                                                        heads=heads[2],
                                                        gamma = gamma)

        self.transformer_block_refinement = TransformerBlock(in_dimensions= in_dimensions[1],
                                                             layers= 4,
                                                             heads=1,
                                                             gamma = gamma)

        self.upsample_1 = nn.Sequential(nn.Conv2d(in_channels= in_dimensions[-1],
                                                   out_channels= in_dimensions[-1] * 2,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros",
                                                   bias=False),
                                         nn.PixelShuffle(upscale_factor= 2))
        self.upsample_2 = nn.Sequential(nn.Conv2d(in_channels= in_dimensions[-2],
                                                   out_channels= in_dimensions[-2] * 2,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros",
                                                   bias=False),
                                         nn.PixelShuffle(upscale_factor= 2))
        self.upsample_3 = nn.Sequential(nn.Conv2d(in_channels= in_dimensions[-3],
                                                   out_channels= in_dimensions[-3] * 2,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros",
                                                   bias=False),
                                         nn.PixelShuffle(upscale_factor= 2))

        self.concat_reduction_1 = nn.Conv2d(in_channels= in_dimensions[-1],
                                            out_channels= in_dimensions[-1] // 2,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias= False)

        self.concat_reduction_2 = nn.Conv2d(in_channels= in_dimensions[-2],
                                            out_channels= in_dimensions[-2] // 2,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias= False)

        self.residual_reconstruction = nn.Conv2d(in_channels=in_dimensions[1],
                                                 out_channels= out_channels,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 padding_mode="zeros",
                                                 bias= False)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x_in = x
        x = self.shallow_features(x)
        residual_1 = self.transformer_block_down_1(x)
        # print(residual_1.shape)
        x = self.downsample_1(residual_1)
        # print(x.shape)
        residual_2 = self.transformer_block_down_2(x)
        # print(residual_2.shape)
        x = self.downsample_2(residual_2)
        # print(x.shape)
        residual_3 = self.transformer_block_down_3(x)
        # print(residual_3.shape)
        x = self.downsample_3(residual_3)
        # print(x.shape)
        x = self.transformer_block_4(x)
        # print(x.shape)
        x = self.upsample_1(x)
        # print(x.shape)
        concat_1 = torch.cat((x, residual_3), dim= 1)
        x = self.concat_reduction_1(concat_1)
        # print(x.shape)
        x = self.transformer_block_up_3(x)
        # print(x.shape)
        x = self.upsample_2(x)
        # print(x.shape)
        concat_2 = torch.cat((x, residual_2), dim= 1)
        x = self.concat_reduction_2(concat_2)
        # print(x.shape)
        x = self.transformer_block_up_2(x)
        # print(x.shape)
        x = self.upsample_3(x)
        # print(x.shape)
        concat_3 = torch.cat((x, residual_1), dim= 1)
        # print(concat_3.shape)
        x = self.transformer_block_up_1(concat_3)
        # print(x.shape)
        x = self.transformer_block_refinement(x)
        # print(x.shape)
        x = self.residual_reconstruction(x)
        # print(x.shape)
        i_tilda = x + x_in

        return i_tilda
