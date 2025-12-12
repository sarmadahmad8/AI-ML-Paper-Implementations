
import torch
import torch.nn as nn
from typing import Tuple, List


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int):
        super().__init__()

        self.patcher = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding_mode='zeros',
                                    padding=padding)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:

        # print(x.shape)
        x = self.patcher(x)
        # print(x.shape)
        x = x.reshape([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]).permute(0, 2, 1)
        return x

class EfficientSelfAttention(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 reduction: int,
                 num_heads: int,
                 attn_dropout: float):
        
        super().__init__()
        self.reduction = reduction
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.esa = nn.MultiheadAttention(embed_dim= embedding_dim,
                                        num_heads= num_heads,
                                        dropout= attn_dropout,
                                        batch_first= True)

        self.linear_reduction = nn.Linear(in_features=embedding_dim * reduction,
                                        out_features=embedding_dim)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        # print(x.shape)
        k_hat = x.reshape([x.shape[0], x.shape[1] // self.reduction, x.shape[2] * self.reduction])
        # print(k_hat.shape)
        k = self.linear_reduction(k_hat)
        # print(k.shape)
        attn_output, _ = self.esa(query = x,
                                    key = k,
                                    value = k,
                                    need_weights= False)
        return attn_output

class MixFFN(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 expansion_factor: int,
                 dropout: float = 0.1):
        super().__init__()

        self.mlp_1 = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                nn.Linear(in_features=embedding_dim,
                                            out_features=embedding_dim * expansion_factor)
                                  )
        self.pos_embed_conv = nn.Conv2d(in_channels=embedding_dim * expansion_factor,
                                        out_channels=embedding_dim * expansion_factor,
                                        kernel_size=3,
                                        stride=1,
                                        padding_mode='zeros',
                                        padding=1,
                                        bias=True,
                                        groups=embedding_dim * expansion_factor
                                       )
        self.mlp_2 = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim * expansion_factor),
                                nn.Linear(in_features=embedding_dim * expansion_factor,
                                            out_features=embedding_dim)
                                  )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout,
                                  inplace=True)

    def forward(self,
                x_in: torch.Tensor) -> torch.Tensor:
        x = self.mlp_1(x_in)
        # print(x.shape)
        x = x.permute(2, 0, 1)
        # print(x.shape)
        x = self.pos_embed_conv(x)
        # print(x.shape)
        x = x.permute(1, 2, 0)
        # print(x.shape)
        x = self.gelu(x)
        # print(x.shape)
        x = self.dropout(x)
        x = self.mlp_2(x)
        # print(x.shape)
        x = self.dropout(x)
        return x + x_in

class TransformerBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 reduction: int,
                 num_heads: int,
                 attn_dropout: float,
                 expansion_factor: int,
                 num_layers: int,
                 dropout: float = 0.1):
        
        super().__init__()

        self.transformer_layer = nn.Sequential(EfficientSelfAttention(embedding_dim=embedding_dim,
                                                                        reduction=reduction,
                                                                        num_heads= num_heads,
                                                                        attn_dropout= attn_dropout),
                                               MixFFN(embedding_dim= embedding_dim,
                                                                        expansion_factor= expansion_factor,
                                                                        dropout= dropout)
                                              )
        self.transformer_block = nn.Sequential(*[self.transformer_layer for num in range(num_layers)])

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        return self.transformer_block(x)

class SegFormerEncoder(nn.Module):
    def __init__(self,
                 height: Tuple[int, int, int, int],
                 width: Tuple[int, int, int, int],
                 in_channels: Tuple[int, int, int, int],
                 out_channels: Tuple[int, int, int, int], 
                 kernel_size: Tuple[int, int, int, int],
                 stride: Tuple[int, int, int, int],
                 padding: Tuple[int, int, int, int],
                 reduction: Tuple[int, int, int, int],
                 num_heads: Tuple[ int, int, int, int],
                 expansion_factor: Tuple[int, int, int, int],
                 num_layers: Tuple[int, int, int, int],
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0):

        super().__init__()

        self.height = height
        self.width = width
        self.reduction = ((height[0] * 4) // (reduction[0]), (height[0] * 4) // (reduction[0] * reduction[1]), (height[0] * 4) // (reduction[0] * reduction[1] * reduction[2]), (height[0] * 4) // (reduction[0] * reduction[1] * reduction[2])* reduction[3])
        self.encoder_layer = nn.ModuleList()

        for idx, channel in enumerate(in_channels):
            self.encoder_layer.append(nn.Sequential(PatchEmbedding(in_channels=in_channels[idx],
                                                     out_channels=out_channels[idx],
                                                     kernel_size= kernel_size[idx],
                                                     stride= stride[idx],
                                                     padding= padding[idx]),
                                     TransformerBlock(embedding_dim=out_channels[idx],
                                                        num_heads= num_heads[idx],
                                                        reduction= self.reduction[idx],
                                                        expansion_factor= expansion_factor[idx],
                                                        num_layers= num_layers[idx],
                                                        attn_dropout= attn_dropout,
                                                        dropout= dropout)
                                                   )
                                     )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        residual_connections = []
        for i, layer_block in enumerate(self.encoder_layer):
            x = layer_block(x)
            x = x.reshape([x.shape[0], self.height[i], self.width[i], x.shape[2]]).permute(0, 3, 1, 2)
            # print(f"After {i+1} transformer block: {x.shape}")
            residual_connections.append(x)
        return residual_connections

class SegFormerDecoder(nn.Module):
    def __init__(self,
                 scale: Tuple[ int, int, int, int],
                 unified_channel_size: int,
                 out_channels: Tuple[int, int, int, int],
                 num_classes: int,
                dropout: float = 0.1):

        super().__init__()
        self.scale = scale
        
        self.mlp_block = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for idx, channel in enumerate(out_channels):
            self.mlp_block.append(nn.Linear(in_features=channel,
                                                    out_features= unified_channel_size)
                           )
            self.upsample.append(nn.Upsample(scale_factor= scale[idx],
                                                    mode= 'bilinear')
                                )


        self.mlp_layer = nn.Linear(in_features=4 * unified_channel_size,
                                                         out_features= unified_channel_size)
        self.linear_projection = nn.Linear(in_features= unified_channel_size,
                                            out_features= num_classes)

    def forward(self,
                x: List[torch.Tensor]) -> torch.Tensor:

        upsampled_features = []
        for idx, module in enumerate(self.mlp_block):
            # print(x[idx].shape)
            x[idx] = x[idx].permute(0, 2, 3, 1)
            # print(x[idx].shape)
            f_hat = module(x[idx])
            # print(f_hat.shape)
            f_hat = f_hat.permute(0, 3, 1, 2)
            f_hat = self.upsample[idx](f_hat)
            upsampled_features.append(f_hat)
            # print(f_hat.shape)

        f_hat_concat = torch.cat(upsampled_features, dim=1)
        # print(f_hat_concat.shape)
        f_hat_concat = f_hat_concat.permute(0, 2, 3, 1)
        f = self.mlp_layer(f_hat_concat)
        # print(f.shape)
        m = self.linear_projection(f)
        m = m.permute(0, 3, 1, 2)
        # print(m.shape)
        return m

class SegFormer(nn.Module):
    def __init__(self,
                 height: Tuple[int, int, int, int],
                 width: Tuple[int, int, int, int],
                 in_channels: Tuple[int, int, int, int],
                 out_channels: Tuple[int, int, int, int], 
                 kernel_size: Tuple[int, int, int, int],
                 stride: Tuple[int, int, int, int],
                 padding: Tuple[int, int, int, int],
                 reduction: Tuple[int, int, int, int],
                 num_heads: Tuple[ int, int, int, int],
                 expansion_factor: Tuple[int, int, int, int],
                 num_layers: Tuple[int, int, int, int],
                 scale: Tuple[ int, int, int, int],
                 unified_channel_size: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0):

        super().__init__()

        self.segformer_encoder = SegFormerEncoder(height= height,
                                                 width= width,
                                                 in_channels= in_channels,
                                                 out_channels= out_channels,
                                                 kernel_size= kernel_size,
                                                 stride= stride,
                                                 padding= padding,
                                                 reduction= reduction,
                                                 num_heads= num_heads,
                                                 expansion_factor= expansion_factor,
                                                 num_layers= num_layers
                                                 )

        self. segformer_decoder = SegFormerDecoder(scale = scale,
                                                   out_channels= out_channels,
                                                   unified_channel_size= unified_channel_size,
                                                   num_classes= num_classes,
                                                  dropout= dropout)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:

        return self.segformer_decoder(self.segformer_encoder(x))
