import torch
import torch.nn as nn

class LinearNoiseScheduler:

    def __init__(self,
                 num_timesteps: int,
                 beta_start: float,
                 beta_end: float,
                 device: torch.device):

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)

    def add_noise(self,
                  original: torch.Tensor,
                  noise: torch.Tensor,
                  t: int):

        original_shape = original.shape
        batch_size = original.shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(dim=-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(dim=-1)

        # sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.expand(batch_size, 1, 1 , 1)
        # sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod(batch_size, 1, 1, 1)
        
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    def sample_prev_timestep(self,
                             xt: torch.Tensor,
                             noise_pred: torch.Tensor,
                             t: int):

        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, min=-1.0, max=1.0)

        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0

        else:
            variance = ((self.betas[t]) * (1.0 - self.alpha_cum_prod[t-1])) / (1.0 - self.alpha_cum_prod[t])
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0

def get_time_embedding(time_steps: torch.Tensor,
                       t_embed_dim: int):

    factor = 10000 ** (torch.arange(start=0, end=t_embed_dim // 2, device=time_steps.device) / (t_embed_dim // 2))
    t_emb = time_steps[:, None].repeat(1, t_embed_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

    return t_emb

class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_embed_dim: int,
                 down_sample: bool,
                 num_heads: int):
        super().__init__()

        self.conv_first = nn.Sequential(nn.GroupNorm(num_groups=8,
                                                     num_channels=in_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  padding_mode="zeros"))

        self.t_embed_layers = nn.Sequential(nn.SiLU(),
                                            nn.Linear(in_features=t_embed_dim,
                                                      out_features=out_channels))

        self.conv_second = nn.Sequential(nn.GroupNorm(num_groups=8,
                                                      num_channels=out_channels),
                                         nn.SiLU(),
                                         nn.Conv2d(in_channels=out_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros"))

        self.attention_norm = nn.GroupNorm(num_groups=8,
                                           num_channels=out_channels)

        self.attention = nn.MultiheadAttention(embed_dim=out_channels,
                                               num_heads=num_heads,
                                               batch_first=True)

        self.res_input = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.downsample = nn.AvgPool2d(kernel_size=2,
                                       stride=2) if down_sample else nn.Identity()

    def forward(self,
                x_in: torch.Tensor,
                t_embed: torch.Tensor):

        x = self.conv_first(x_in)
        x = x + self.t_embed_layers(t_embed)[:, :, None, None]
        x = self.conv_second(x)
        x = x + self.res_input(x_in)

        B, C, H, W = x.shape
        in_attn = x.reshape(B, C, H * W)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(query=in_attn,
                                     key=in_attn,
                                     value=in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(B, C, H, W)
        x = x + out_attn

        x = self.downsample(x)

        return x

class BottleNeckBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_embed_dim: int,
                 num_heads: int):

        super().__init__()

        self.conv_first = nn.ModuleList([nn.Sequential(nn.GroupNorm(num_groups=8,
                                                                    num_channels=in_channels),
                                                       nn.SiLU(),
                                                       nn.Conv2d(in_channels=in_channels,
                                                                 out_channels=out_channels,
                                                                 kernel_size=3,
                                                                 stride=1,
                                                                 padding=1,
                                                                 padding_mode="zeros")),
                                          nn.Sequential(nn.GroupNorm(num_groups=8,
                                                                    num_channels=out_channels),
                                                       nn.SiLU(),
                                                       nn.Conv2d(in_channels=out_channels,
                                                                 out_channels=out_channels,
                                                                 kernel_size=3,
                                                                 stride=1,
                                                                 padding=1,
                                                                 padding_mode="zeros"))])

        self.t_embed_layers = nn.ModuleList([nn.Sequential(nn.SiLU(),
                                                           nn.Linear(in_features=t_embed_dim,
                                                                     out_features=out_channels)),
                                             nn.Sequential(nn.SiLU(),
                                                           nn.Linear(in_features=t_embed_dim,
                                                                     out_features=out_channels))])

        self.conv_second = nn.ModuleList([nn.Sequential(nn.GroupNorm(num_groups=8,
                                                                     num_channels=out_channels),
                                                        nn.SiLU(),
                                                        nn.Conv2d(in_channels=out_channels,
                                                                  out_channels=out_channels,
                                                                  kernel_size=3,
                                                                  stride=1,
                                                                  padding=1,
                                                                  padding_mode="zeros")),
                                           nn.Sequential(nn.GroupNorm(num_groups=8,
                                                                     num_channels=out_channels),
                                                        nn.SiLU(),
                                                        nn.Conv2d(in_channels=out_channels,
                                                                  out_channels=out_channels,
                                                                  kernel_size=3,
                                                                  stride=1,
                                                                  padding=1,
                                                                  padding_mode="zeros"))])

        self.attention_norm = nn.GroupNorm(num_groups=8,
                                           num_channels=out_channels)

        self.attention = nn.MultiheadAttention(embed_dim=out_channels,
                                               num_heads=num_heads,
                                               batch_first=True)

        self.res_input = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0),
                                         nn.Conv2d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0)])

    def forward(self,
                x_in: torch.Tensor,
                t_embed: torch.Tensor):

        x = self.conv_first[0](x_in)
        x = x + self.t_embed_layers[0](t_embed)[:, :, None, None]
        x = self.conv_second[0](x)
        x = x + self.res_input[0](x_in)

        B, C, H, W = x.shape
        in_attn = x.reshape(B, C, H * W)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(query=in_attn,
                                     key=in_attn,
                                     value=in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(B, C, H, W)
        x = x + out_attn

        x_in = x
        x = self.conv_first[1](x_in)
        x = x + self.t_embed_layers[1](t_embed)[:, :, None, None]
        x = self.conv_second[1](x)
        x = x + self.res_input[1](x_in)

        return x

class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 t_embed_dim: int,
                 up_sample: bool,
                 num_heads: int):
        super().__init__()

        self.conv_first = nn.Sequential(nn.GroupNorm(num_groups=8,
                                                     num_channels=in_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  padding_mode="zeros"))

        self.t_embed_layers = nn.Sequential(nn.SiLU(),
                                            nn.Linear(in_features=t_embed_dim,
                                                      out_features=out_channels))

        self.conv_second = nn.Sequential(nn.GroupNorm(num_groups=8,
                                                      num_channels=out_channels),
                                         nn.SiLU(),
                                         nn.Conv2d(in_channels=out_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros"))

        self.attention_norm = nn.GroupNorm(num_groups=8,
                                           num_channels=out_channels)

        self.attention = nn.MultiheadAttention(embed_dim=out_channels,
                                               num_heads=num_heads,
                                               batch_first=True)

        self.res_input = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.upsample = nn.ConvTranspose2d(in_channels=in_channels // 2,
                                             out_channels= in_channels // 2,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1,
                                             padding_mode="zeros") if up_sample else nn.Identity()

    def forward(self,
                x_in: torch.Tensor,
                out_down: torch.Tensor,
                t_embed: torch.Tensor):
        x_in = self.upsample(x_in)
        x_in = torch.cat([x_in, out_down], dim=1)

        x = self.conv_first(x_in)
        x = x + self.t_embed_layers(t_embed)[:, :, None, None]
        x = self.conv_second(x)
        x = x + self.res_input(x_in)

        B, C, H, W = x.shape
        in_attn = x.reshape(B, C, H * W)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(query=in_attn,
                                     key=in_attn,
                                     value=in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(B, C, H, W)
        x = x + out_attn

        return x

class Unet(nn.Module):

    def __init__(self,
                in_channels: int):

        super().__init__()

        self.down_channels = [64, 128, 256, 512]
        self.mid_channels = [512, 256, 256]
        self.t_embed_dim = 128
        self.down_sample = [True, True, False]

        self.t_projection = nn.Sequential(nn.Linear(in_features=self.t_embed_dim,
                                                    out_features=self.t_embed_dim),
                                          nn.SiLU(),
                                          nn.Linear(in_features=self.t_embed_dim,
                                                    out_features=self.t_embed_dim))

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=self.down_channels[0],
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 padding_mode="zeros")

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(in_channels=self.down_channels[i],
                                        out_channels=self.down_channels[i+1],
                                        t_embed_dim=self.t_embed_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=4))
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) -1):
            self.mids.append(BottleNeckBlock(in_channels=self.mid_channels[i],
                                             out_channels=self.mid_channels[i+1],
                                             t_embed_dim=self.t_embed_dim,
                                             num_heads=4))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(in_channels=self.down_channels[i] * 2,
                                    out_channels=self.down_channels[i-1] if i !=0 else 16,
                                    t_embed_dim=self.t_embed_dim,
                                    up_sample=self.down_sample[i],
                                    num_heads=4))

        self.norm_out = nn.GroupNorm(num_groups=8,
                                     num_channels=16)
        self.conv_out = nn.Conv2d(in_channels=16,
                                  out_channels=in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  padding_mode="zeros")

    def forward(self,
                x_in: torch.Tensor,
                timesteps: int):

        out = self.conv_in(x_in)
        t_embed = get_time_embedding(time_steps=timesteps,
                                     t_embed_dim=self.t_embed_dim)
        t_embed = self.t_projection(t_embed)

        down_outs = []
        for down in self.downs:
            # print(out.shape)
            down_outs.append(out)
            out = down(out, t_embed)

        for mid in self.mids:
            # print(out.shape)
            out = mid(out, t_embed)

        for up in self.ups:
            down_out = down_outs.pop()
            # print(out.shape, down_out.shape)
            out = up(out, down_out, t_embed)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)

        return out
