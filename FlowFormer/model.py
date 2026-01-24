import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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
                                    kernel_size=1,
                                    stride=2,
                                    padding=0,
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

class PatchedCostVolume(nn.Module):

    def __init__(self):

        super().__init__()

        self.patcher = nn.Sequential(nn.Conv2d(in_channels=1,
                                               out_channels=16,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               padding_mode="zeros"),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels=16,
                                               out_channels=32,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               padding_mode="zeros"),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels=32,
                                               out_channels=64,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               padding_mode="zeros"),
                                     nn.ReLU(inplace=True))

    def _contruct_cost_volume(self,
                              fmap_1: torch.Tensor,
                              fmap_2: torch.Tensor):

        B, C, H, W = fmap_1.shape

        fmap_1 = F.normalize(fmap_1, dim=1, p=2)  # L2 normalize
        fmap_2 = F.normalize(fmap_2, dim=1, p=2)

        cost_volume = torch.einsum('bchw, bcHW -> bhwHW', fmap_1, fmap_2)
        
        return cost_volume.view(B * H * W, 1, H, W)

    def forward(self,
                fmap_1: torch.Tensor,
                fmap_2: torch.Tensor) -> torch.Tensor:

        cost_volume = self._contruct_cost_volume(fmap_1=fmap_1,
                                                 fmap_2=fmap_2)

        # print(cost_volume.shape)

        patched_cv = self.patcher(cost_volume)
        
        # print(patched_cv.shape)
        
        return patched_cv, cost_volume

class PatchTokenizer(nn.Module):

    def __init__(self,
                 batch: int,
                 img_height: int,
                 img_width: int,
                 K: int,
                 D: int):

        super().__init__()
        self.batch = batch
        self.feat_height = img_height // 8
        self.feat_width = img_width // 8
        self.pe_height, self.pe_width = self.feat_height // 8, self.feat_width // 8
        
        self.position_embedding = nn.Parameter(torch.randn([64, self.pe_height, self.pe_width],
                                                            dtype=torch.float32),
                                               requires_grad=True)

        self.latent_codewords = nn.Parameter(torch.randn([K, D], dtype=torch.float32),
                                             requires_grad=True)

        self.kv_conv = nn.Conv2d(in_channels=128,
                                 out_channels=128 * 2,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

    def forward(self,
                F_x: torch.Tensor) -> torch.Tensor:

        position_embedding = self.position_embedding.unsqueeze(dim=0).expand(F_x.shape[0], -1, -1, -1)
        # print(position_embedding.shape)
        K_x, V_x = self.kv_conv(torch.cat([F_x, position_embedding], dim=1)).chunk(2, dim=1) # [B * H * W, 2 * D_p, H // 8, W // 8]
        # print(K_x.shape, V_x.shape)  # [B * H * W, 2 * D_p, H // 8, W // 8]

        ck_dot = torch.einsum('kd, bdhw -> bkhw', self.latent_codewords, K_x) # [B * H * W, D, H // 8, W // 8]

        ck_dot = ck_dot * (ck_dot.shape[1] ** -0.5)
        
        score = torch.softmax(ck_dot, dim=1)
        
        T_x = torch.einsum('bkhw, bdhw -> bkd', ck_dot, V_x) # [B * H * W, D, K]

        return T_x


class AlternateGroupTransformerLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_expansion: int,
                 batch: int,
                 img_height: int,
                 img_width: int):

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch = batch
        self.feat_height = img_height // 8
        self.feat_width = img_width // 8

        self.intra_cost_qkv = nn.Linear(in_features=embed_dim,
                                        out_features=embed_dim * 3)
        
        self.intra_cost_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                     num_heads=num_heads,
                                                     batch_first=True)

        self.inter_cost_qk = nn.Linear(in_features=embed_dim * 2,
                                       out_features=embed_dim * 2)
        self.inter_cost_v = nn.Linear(in_features=embed_dim,
                                      out_features=embed_dim)

        self.intra_ffn = nn.Sequential(nn.Linear(in_features=embed_dim,
                                                 out_features=embed_dim * mlp_expansion),
                                       nn.GELU(),
                                       nn.Linear(in_features=embed_dim * mlp_expansion,
                                                 out_features= embed_dim))

        self.inter_ffn = nn.Sequential(nn.Linear(in_features=embed_dim,
                                                 out_features=embed_dim * mlp_expansion),
                                       nn.GELU(),
                                       nn.Linear(in_features=embed_dim * mlp_expansion,
                                                 out_features= embed_dim))

        self.intra_layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.intra_layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.inter_layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.inter_layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim)

        self.context_projection = nn.Linear(in_features= embed_dim,
                                            out_features=embed_dim)

    def inter_cost_attn(self,
                        K: int,
                        D: int,
                        qk_input: torch.Tensor,
                        v_input: torch.Tensor):
        
        q_x, k_x = self.inter_cost_qk(qk_input).chunk(2, dim=-1)
        # print(q_x.shape, k_x.shape)
        v_x = self.inter_cost_v(v_input)
        # print(v_x.shape)
        q_x = q_x.reshape(self.batch * K, self.feat_height * self.feat_width, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_x = k_x.reshape(self.batch * K, self.feat_height * self.feat_width, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_x = v_x.reshape(self.batch * K, self.feat_height * self.feat_width, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.MATH,  # Fallback
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,  # Best performance
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):# Memory efficient
            
            inter_attn_out = F.scaled_dot_product_attention(
                q_x, k_x, v_x,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )

        return inter_attn_out.reshape(self.batch * K, self.feat_height * self.feat_width, D)
        
    def forward(self,
                T_x: torch.Tensor,
                t_x: torch.Tensor):

        context = t_x
        
        _, K, D = T_x.shape
        q, k, v = self.intra_cost_qkv(T_x).chunk(3, dim=-1)
        
        intra_attn_out, _ = self.intra_cost_attn(query=q,
                                                 key=k,
                                                 value=v)
        x = self.intra_layernorm_1(T_x + intra_attn_out)

        ffn_out = self.intra_ffn(x)
        x = self.intra_layernorm_2(x + ffn_out)

        x = x.view(self.batch * K, self.feat_height * self.feat_width, D)
        # print(x.shape)
        # print(t_x.shape)
        t_x = t_x.reshape(self.batch * self.feat_height * self.feat_width, -1)
        # print(t_x.shape)
        c_t_x = self.context_projection(t_x).unsqueeze(dim=1).expand(-1, K, -1).reshape(self.batch * K, self.feat_height * self.feat_width, D)
        # print(c_t_x.shape)
        attn_input = torch.cat([x, c_t_x], dim=2)
        # print(attn_input.shape)
        inter_attn_out = self.inter_cost_attn(qk_input=attn_input,
                                              v_input=x,
                                              K= K,
                                              D= D)
        x = self.inter_layernorm_1(x + inter_attn_out)

        ffn_out = self.inter_ffn(x)

        x = self.inter_layernorm_2(x + ffn_out)

        return x.view(self.batch * self.feat_height * self.feat_width, K, D), context


class Encoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_expansion: int,
                 img_height: int,
                 img_width: int,
                 batch: int):

        super().__init__()
        self.batch = batch
        self.feat_height = img_height // 4
        self.feat_width = img_width // 4
        
        
        self.feature_extractor = FeatureExtractor(in_channels=3)
        self.context_network = FeatureExtractor(in_channels=3)
        self.patched_cost_volume = PatchedCostVolume()
        self.patch_tokenizer = PatchTokenizer(img_height=img_height,
                                              img_width=img_width,
                                              batch=batch,
                                              K=8,
                                              D=128)
        
        self.agt_1 = AlternateGroupTransformerLayer(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    mlp_expansion=mlp_expansion,
                                                    batch=batch,
                                                    img_height=img_height,
                                                    img_width=img_width)

        self.agt_2 = AlternateGroupTransformerLayer(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    mlp_expansion=mlp_expansion,
                                                    batch=batch,
                                                    img_height=img_height,
                                                    img_width=img_width)

        self.agt_3 = AlternateGroupTransformerLayer(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    mlp_expansion=mlp_expansion,
                                                    batch=batch,
                                                    img_height=img_height,
                                                    img_width=img_width)

    def forward(self,
                both_img: torch.Tensor) -> torch.Tensor:
        
        img_1, img_2 = both_img.chunk(2, dim=1)
        
        fmap_1, fmap_2 = self.feature_extractor(img_1), self.feature_extractor(img_2)

        cnet_out = self.context_network(img_1)

        context_feat, hidden_state = cnet_out.chunk(2, dim=1)

        context_feat, hidden_state = F.relu(context_feat), F.tanh(hidden_state)
        
        patched_cost_volume, cost_volume = self.patched_cost_volume(fmap_1, fmap_2)
        
        T_x = self.patch_tokenizer(patched_cost_volume)

        cost_memory, context_feat = self.agt_3(*self.agt_2(*self.agt_1(T_x, context_feat)))

        return cost_volume, cost_memory, context_feat, hidden_state

class CostMemoryAggregation(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 mlp_expansion: int,
                 num_heads: int,
                 batch: int,
                 img_height: int,
                 img_width: int,
                 local_radius: int):

        super().__init__()
        self.batch = batch
        self.feat_height = img_height // 8
        self.feat_width = img_width // 8
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.local_radius = local_radius
        self.local_channels = (2 * local_radius + 1) ** 2
        
        self.ffn = nn.Sequential(nn.Linear(in_features=embed_dim,
                                           out_features=embed_dim * mlp_expansion),
                                   nn.GELU(),
                                   nn.Linear(in_features=embed_dim * mlp_expansion,
                                               out_features= embed_dim))

        self.flow_token_encoder = nn.Sequential(nn.Conv2d(in_channels=self.local_channels,
                                                          out_channels=embed_dim,
                                                          kernel_size=1,
                                                          stride=1),
                                                nn.GELU(),
                                                nn.Conv2d(in_channels=embed_dim,
                                                          out_channels=embed_dim,
                                                          kernel_size=1,
                                                          stride=1))

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=self.num_heads,
                                                    batch_first=True)

        self.patch_embeddings = nn.Parameter(torch.randn([embed_dim, self.feat_height, self.feat_width],
                                                         dtype=torch.float32),
                                             requires_grad=True)
    def _warp(self,
              img: torch.Tensor,
              coords: torch.Tensor,
              mode: str = "bilinear",
              mask: bool = False):

        H, W = img.shape[-2:]
        x_grid, y_grid = coords.split([1, 1], dim= -1)

        x_grid, y_grid = 2*x_grid/(W-1)-1, 2*y_grid/(H-1)-1

        grid = torch.cat([x_grid, y_grid], dim=-1)
        # print(img.shape)
        # print(grid.shape)
        img = F.grid_sample(input=img,
                            grid=grid,
                            align_corners=True,
                            mode=mode)

        if mask:
            mask = (x_grid > -1) & (y_grid > -1) & (x_grid < 1) & (y_grid < 1)
            return img, mask.float()

        return img
        
    def _compute_corr(self,
                      corr: torch.Tensor,
                      coords: torch.Tensor):

        r = self.local_radius
        B, _, H, W = coords.shape
        coords = coords.permute(0, 2, 3, 1)

        dy = torch.arange(-r, r+1, device=coords.device)
        dx = torch.arange(-r, r+1, device=coords.device)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), dim=-1)

        coords_scaled = coords
        
        coords_scaled = coords_scaled.reshape(B * H * W, 1, 1, 2)
        delta = delta.reshape(1, 2*r+1, 2*r+1, 2)
        # print(delta.shape, coords_scaled.shape)
        sample_coords = coords_scaled + delta
        # print(sample_coords.shape, corr.shape)
        corr_sampled = self._warp(img=corr,
                                  coords=sample_coords,
                                  mode="bilinear")
        # print(corr_sampled.shape)
        corr_sampled = corr_sampled.reshape(B, H, W, -1)
        # print(corr_sampled.shape)
        return corr_sampled.permute(0, 3, 1, 2)

    def forward(self,
                coords: torch.Tensor,
                cost_volume: torch.Tensor,
                T_x: torch.Tensor) -> torch.Tensor:

        
        local_cost = self._compute_corr(coords=coords,
                                        corr=cost_volume) # [B, 81, H, W]

        # print(local_cost.shape)
        q_x = self.flow_token_encoder(local_cost)
        B, C, H, W = q_x.shape
        # print(q_x.shape)
        Q_x = self.ffn((q_x + self.patch_embeddings.unsqueeze(dim=0)).reshape(B, H, W, -1))

        # print(T_x.shape)
        K_x = self.ffn(T_x.view(B * H * W , 8, -1))
        V_x = self.ffn(T_x.view(B * H * W, 8,  -1))

        Q_x = Q_x.view(B * H * W, 1, -1)
        K_x = K_x.view(B * H * W, self.num_heads, -1)
        V_x = V_x.view(B * H * W, self.num_heads, -1)

        # print(Q_x.shape, K_x.shape, V_x.shape)
        c_x, _ = self.multihead_attn(query=Q_x,
                                     key=K_x,
                                     value=V_x)

        return c_x, q_x

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

class BasicMotionEncoder(nn.Module):

    def __init__(self,
                 in_channels: int):

        super().__init__()

        self.conv_corr = nn.Sequential(nn.Conv2d(in_channels=in_channels * 2,
                                                 out_channels=256,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding="same",
                                                 padding_mode="zeros"),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels=256,
                                                 out_channels=192,
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

        self.conv_corr_flow = nn.Sequential(nn.Conv2d(in_channels=192 + 64,
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

class Aggregate(nn.Module):

    def __init__(self,
                 heads: int):

        super().__init__()
        self.heads = heads
        self.gamma = nn.Parameter(torch.zeros(1),
                                  requires_grad=True)

    def forward(self,
                motion_feat: torch.Tensor,
                score: torch.Tensor):

        B, C, H, W = motion_feat.shape
        heads = self.heads
        head_dim = C // heads
        
        v = motion_feat.reshape(B, heads, H * W, head_dim)
        
        score = score.reshape(B, heads, H * W, H * W).contiguous()
        
        attn_out = torch.einsum('bhij, bhik -> bhjk', score, v)
        
        attn_out = attn_out.transpose(2, 3).reshape(B, C, H, W)
        
        return motion_feat.add_(self.gamma * attn_out)

class UpdateBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.motion_encoder = BasicMotionEncoder(in_channels=128)
        self.gru = ConvGRUPair(in_channels=384,
                               kernel_size_1=(1, 5),
                               kernel_size_2=(5, 1))
        self.aggregate = Aggregate(heads=8)

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
                corr: torch.Tensor,
                attn: torch.Tensor = None):

        motion_feat = self.motion_encoder(corr, delta_flow)
        # print(motion_feat.shape, con_net_in.shape)
        if attn is not None:
            motion_feat = self.aggregate(motion_feat, attn)
            
        gru_in = torch.cat([con_net_in, motion_feat], dim=1)

        hidden_state_new = self.gru(hidden_state, gru_in)
        delta_flow_new = self.flow_head(hidden_state_new)

        upsampled_flow_mask = 0.25 * self.upsample_block(hidden_state_new)

        return hidden_state_new, delta_flow_new, upsampled_flow_mask

class AttendContext(nn.Module):

    def __init__(self,
                 heads: int):

        super().__init__()
        self.heads = heads

    def forward(self,
                context_map: torch.Tensor) -> torch.Tensor:

        B, C, H, W = context_map.shape
        heads = self.heads
        head_dim = C // self.heads
        
        q = rearrange(context_map, 'b (h d) H W -> b h (H W) d', b=B, h=heads, d=head_dim, H=H, W=W)
        k = q
        # print(k.shape, q.shape)
        qk_dot = torch.einsum('bhik, bhjk -> bhij', q, k)

        score = torch.softmax((qk_dot * (head_dim ** -0.5)), dim=-1)

        return score

class Decoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_expansion: int,
                 img_height: int,
                 img_width: int,
                 batch: int,
                 local_radius: int,
                 iters: int):

        super().__init__()
        self.batch = batch
        self.feat_height = img_height // 4
        self.feat_width = img_width // 4
        self.iters = iters

        self.update_block = UpdateBlock()
        self.attend_context = AttendContext(heads=num_heads)
        self.cost_memory_agg = CostMemoryAggregation(embed_dim=embed_dim,
                                                     num_heads=num_heads,
                                                     mlp_expansion=mlp_expansion,
                                                     batch=batch,
                                                     img_height=img_height,
                                                     img_width=img_width,
                                                     local_radius=local_radius)

    def _init_flow(self,
                   img: torch.Tensor):

        B, C, H, W = img.shape

        init_grid = torch.meshgrid([torch.arange(H, device= img.device), torch.arange(W, device= img.device)], indexing='ij')
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
                cost_volume: torch.Tensor,
                cost_memory: torch.Tensor,
                context_feat: torch.Tensor,
                hidden_state: torch.Tensor):
        
        coords0, coords1 = self._init_flow(img= context_feat)
        #print(cost_memory.shape, T_x.shape)
        qk_context = self.attend_context(context_feat)
        # print(qk_context.shape)
        flow_predictions = []
        for iter in range(self.iters):
            coords1 = coords1.detach()

            c_x, q_x = self.cost_memory_agg(coords1,
                                            cost_volume,
                                            cost_memory)

            c_x = c_x.view_as(q_x)
            # print(c_x.shape, q_x.shape, context_feat.shape)
            gru_in = torch.cat([q_x, c_x], dim=1)
            
            flow = coords1 - coords0
            
            hidden_state, delta_flow, upsampled_flow_mask = self.update_block(hidden_state, context_feat, flow, gru_in, qk_context)

            # delta_flow = torch.clamp(delta_flow, min=-100, max=100)

            coords1 = coords1 + delta_flow

            flow_up = self._upsample_flow(coords1 - coords0, upsampled_flow_mask)
            # print(flow_up.shape)
            flow_predictions.append(flow_up)

        return flow_predictions

class FlowFormer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_expansion: int,
                 img_height: int,
                 img_width: int,
                 batch: int,
                 local_radius: int,
                 iters: int):

        super().__init__()
        self.batch = batch
        self.feat_height = img_height // 4
        self.feat_width = img_width // 4
        self.iters = iters

        self.encoder = Encoder(embed_dim=embed_dim,
                               num_heads=num_heads,
                               mlp_expansion=mlp_expansion,
                               batch=batch,
                               img_height=img_height,
                               img_width=img_width)

        self.decoder = Decoder(embed_dim=embed_dim,
                               num_heads=num_heads,
                               mlp_expansion=mlp_expansion,
                               batch=batch,
                               img_height=img_height,
                               img_width=img_width,
                               local_radius=local_radius,
                               iters=iters)

    def forward(self,
                both_img = torch.Tensor) -> torch.Tensor:

        enc_out = self.encoder(both_img)
        dec_out = self.decoder(*enc_out)

        return dec_out
