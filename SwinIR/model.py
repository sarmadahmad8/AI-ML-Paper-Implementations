import torch
from torch import nn, einsum
from einops import rearrange
import numpy as np


class ShallowFeatureExtractor(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_dimensions: int):

        super().__init__()

        self.shallow_feature_extractor = nn.Conv2d(in_channels= in_channels,
                                                   out_channels= hidden_dimensions,
                                                   kernel_size= 3,
                                                   stride= 1,
                                                   padding= 1,
                                                   padding_mode='zeros')

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.shallow_feature_extractor(x).permute(0, 2, 3, 1)

class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 expansion_factor: int,
                 dropout: float = 0.1):

        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(in_features= input_dim,
                                           out_features= input_dim * expansion_factor),
                                 nn.GELU(),
                                 nn.Dropout(p= dropout,
                                            inplace=True),
                                 nn.Linear(in_features= input_dim * expansion_factor,
                                           out_features= input_dim),
                                 nn.Dropout(p= dropout,
                                            inplace= True)
                                )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.mlp(x)

class CyclicShift(nn.Module):

    def __init__(self,
                 displacement: int):

        super().__init__()

        self.displacement = displacement

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = torch.roll(input= x,
                       shifts=(self.displacement, self.displacement), 
                       dims= (1, 2)) # [B, H, W, C] -> H = 1, W = 2

        return x

def create_mask(window_size: int,
                displacement: int,
                upper_lower: bool,
                left_right: bool):

    mask = torch.zeros([window_size **2, window_size ** 2]) # mask shifted patches
    # print(mask)
    
    if upper_lower:
        mask[-displacement * window_size: , :-displacement * window_size] = float('-inf') # bottom left section
        mask[: -displacement * window_size, -displacement * window_size:] = float('-inf') # top right section

    if left_right:
        mask = rearrange(mask,
                         '(h1 w1) (h2 w2) -> h1 w1 h2 w2', 
                         h1 = window_size,
                         h2 = window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask,
                         'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
        
    return mask

def get_relative_distances(window_size: int):

    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))

    distances = indices[None, :, :] - indices[:, None, :]

    return distances

class WindowAttention(nn.Module):

    def __init__(self,
                 input_dim: int,
                 heads: int,
                 head_dim: int,
                 shifted: bool,
                 window_size: int,
                 relative_pos_embedding: bool):

        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.scale = heads ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_shift_reverse = CyclicShift(displacement)

            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size,
                                                             displacement=displacement,
                                                             upper_lower=True,
                                                             left_right=False),
                                                 requires_grad=False)
    
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size,
                                                            displacement=displacement,
                                                            upper_lower=False,
                                                            left_right=True),
                                                requires_grad=False)

        self.to_qkv = nn.Linear(in_features= input_dim,
                                out_features= inner_dim * 3, # x3 for Q, K, V
                                bias = False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size= window_size) + window_size -1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size -1, 2 * window_size - 1))

        else:
            self.pos_embedding = nn.Parameter(torch.randn([window_size ** 2,
                                                           window_size ** 2]))
        
        
        self.output = nn.Linear(in_features= inner_dim,
                                out_features= input_dim)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        nw_h, nw_w = n_h // self.window_size, n_w // self.window_size

        q, k, v = map(lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                          h= h, w_h= self.window_size, w_w= self.window_size), qkv)

        q_k_dot_product = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            tmp1 = self.relative_indices[ :, :, 0]

            q_k_dot_product += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]

        else:
            q_k_dot_product += self.pos_embedding

        if self.shifted:
            q_k_dot_product[:, :, -nw_w:] += self.upper_lower_mask # Apply mask only to last row of patch (-nw_w: = last 8 row)
            q_k_dot_product[:, :, nw_w - 1:: nw_w] += self.left_right_mask # Apply mask only to last column of each row of patch (nw_w-1::nw_w = [7, 15, 23, 31, 39, 47, 55])

        score =  q_k_dot_product.softmax(dim= -1)

        attn = einsum('b h w i j, b h w j d -> b h w i d', score, v)

        attn = rearrange(attn, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                         h=h, w_h=self.window_size, w_w= self.window_size, nw_h= nw_h, nw_w=nw_w)

        output = self.output(attn)

        if self.shifted:
            output = self.cyclic_shift_reverse(output)

        return output

class SwinBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 heads: int,
                 head_dim: int,
                 mlp_expansion: int,
                 shifted: bool,
                 window_size: int,
                 relative_pos_embedding: bool):

        super().__init__()

        self.attention_block = WindowAttention(input_dim=input_dim,
                                               heads= heads,
                                               head_dim= head_dim,
                                               shifted= shifted,
                                               window_size= window_size,
                                               relative_pos_embedding= relative_pos_embedding)

        self.mlp = MLP(input_dim=input_dim,
                        expansion_factor= mlp_expansion,
                        dropout= 0.1)

        self.layer_norm = nn.LayerNorm(normalized_shape=input_dim)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        residual_1 = self.attention_block(self.layer_norm(x)) + x
        residual_2 = self.mlp(self.layer_norm(residual_1)) + residual_1

        return residual_2

class ResidualSwinTransformerBlock(nn.Module):

    def __init__(self,
                 hidden_dimension: int,
                 layers: int,
                 mlp_expansion: int,
                 num_heads: int,
                 head_dim: int,
                 window_size: int,
                 relative_pos_embedding: bool):

        super().__init__()
        assert layers % 2 == 0, "Layers have to be divisible by 2 for regular and shifted attention"

        self.residual_swin_transformer_block = nn.ModuleList()

        for _ in range(layers // 2):
            self.residual_swin_transformer_block.append(
                SwinBlock(input_dim= hidden_dimension,
                          heads= num_heads,
                          head_dim= head_dim,
                          mlp_expansion= mlp_expansion,
                          shifted= False,
                          window_size= window_size,
                          relative_pos_embedding= relative_pos_embedding))

            self.residual_swin_transformer_block.append(
                SwinBlock(input_dim= hidden_dimension,
                          heads= num_heads,
                          head_dim= head_dim,
                          mlp_expansion= mlp_expansion,
                          shifted= True,
                          window_size= window_size,
                          relative_pos_embedding= relative_pos_embedding))

        self.rstb_conv = nn.Conv2d(in_channels= hidden_dimension,
                                   out_channels= hidden_dimension,
                                   kernel_size=3,
                                   stride= 1,
                                   padding= 1,
                                   padding_mode='zeros')

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x_in = x
        for swin_transformer_layer in self.residual_swin_transformer_block:
            x = swin_transformer_layer(x)

        x = x.permute(0, 3, 1, 2)
        x = self.rstb_conv(x)
        x = x.permute(0, 2, 3, 1)
        return x + x_in

class DeepFeatureExtraction(nn.Module):

    def __init__(self,
                 rstb_layers: int,
                 mlp_expansion: int,
                 hidden_dimension: int,
                 swin_t_layers: int,
                 num_heads: int,
                 head_dim: int,
                 window_size: int,
                 relative_pos_embedding: bool):

        super().__init__()

        self.rstb_layers = nn.ModuleList()

        for _ in range(rstb_layers):
            self.rstb_layers.append(
                ResidualSwinTransformerBlock(hidden_dimension = hidden_dimension,
                                             mlp_expansion= mlp_expansion,
                                             layers = swin_t_layers,
                                             num_heads = num_heads,
                                             head_dim = head_dim,
                                             window_size = window_size,
                                             relative_pos_embedding = relative_pos_embedding))

        self.deep_feature_extraction_conv = nn.Conv2d(in_channels=hidden_dimension,
                                                      out_channels=hidden_dimension,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1,
                                                      padding_mode='zeros')

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x_in = x.permute(0, 3, 1, 2)
        for rstb in self.rstb_layers:
            x = rstb(x)

        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.deep_feature_extraction_conv(x)

        return x + x_in

class HQImageReconstruction(nn.Module):

    def __init__(self,
                 hidden_dimension: int,
                 scale: int,
                 out_channels: int):

        super().__init__()

        self.upsample_and_reconstruct = nn.Sequential(nn.Conv2d(in_channels= hidden_dimension,
                                                                 out_channels=64,
                                                                 kernel_size= 3,
                                                                 stride=1,
                                                                 padding=1,
                                                                 padding_mode='zeros'),
                                                       nn.Conv2d(in_channels= 64,
                                                                 out_channels= 64 * scale * scale,
                                                                 kernel_size=3,
                                                                 stride=1,
                                                                 padding=1,
                                                                 padding_mode='zeros'),
                                                       nn.PixelShuffle(upscale_factor= scale),
                                                       nn.Conv2d(in_channels= 64,
                                                                 out_channels=out_channels,
                                                                 kernel_size=3,
                                                                 stride=1,
                                                                 padding=1,
                                                                 padding_mode='zeros'))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        x = self.upsample_and_reconstruct(x)

        return x

class SwinIR(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale: int,
                 rstb_layers: int,
                 mlp_expansion: int,
                 hidden_dimension: int,
                 swin_t_layers: int,
                 num_heads: int,
                 head_dim: int,
                 window_size: int,
                 relative_pos_embedding: bool):

        super().__init__()

        self.shallow_feature_extractor = ShallowFeatureExtractor(in_channels=in_channels,
                                                                 hidden_dimensions= hidden_dimension)

        self. deep_feature_extraction = DeepFeatureExtraction(rstb_layers=rstb_layers,
                                                              mlp_expansion= mlp_expansion,
                                                              hidden_dimension= hidden_dimension,
                                                              swin_t_layers= swin_t_layers,
                                                              num_heads= num_heads,
                                                              head_dim= head_dim,
                                                              window_size= window_size,
                                                              relative_pos_embedding= relative_pos_embedding)

        self. hq_image_reconstruction = HQImageReconstruction(hidden_dimension= hidden_dimension,
                                                              scale= scale,
                                                              out_channels= out_channels)

        self.mean = torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        self.mean = self.mean.type_as(x)
        x = (x - self.mean)
        x = self.shallow_feature_extractor(x)
        x = self.deep_feature_extraction(x)
        x = self.hq_image_reconstruction(x)
        x = x + self.mean

        return x
                                                              
                                                                  
