import torch
import torch.nn as nn
from typing import Tuple

class ConvLSTMLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 patch_size: int):

        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.W_x = nn.Conv2d(in_channels= in_channels,
                         out_channels= embed_dim * 4,
                         kernel_size= 5,
                         stride= 1,
                         padding= 2,
                         padding_mode= "zeros",
                         bias= False)

        self.W_h = nn.Conv2d(in_channels= embed_dim,
                         out_channels= embed_dim * 4,
                         kernel_size= 5,
                         stride= 1,
                         padding= 2,
                         padding_mode= "zeros",
                         bias= False)

        self.W_c = nn.Parameter(data= torch.randn((embed_dim * 3, 1, 1)) * 0.01,
                            requires_grad= True)

        self.b = nn.Parameter(data= torch.zeros((embed_dim * 4, 1, 1)),
                           requires_grad= True)

        self.layer_norm = nn.GroupNorm(num_groups= 4,
                                       num_channels= embed_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def _init_state(self,
                    embed_dim: int,
                    patch_size: int,
                    batch_size: int):
        C_t_1 = torch.zeros((batch_size, embed_dim, patch_size, patch_size),
                                  dtype= torch.float32).to(self.W_h.weight.device)
        H_t_1 = torch.zeros((batch_size, embed_dim, patch_size, patch_size),
                                  dtype= torch.float32).to(self.W_h.weight.device)

        return C_t_1, H_t_1
        
    def forward(self,
                X_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        H_t_list = []
        C_t_list = []
        B, T, C, H, W = X_t.shape
        D = self.embed_dim
        
        C_t_1, H_t_1 = self._init_state(embed_dim= D,
                                        patch_size= H,
                                        batch_size= B)

        for frame in range(T):
            W_xi, W_xf, W_xc, W_xo = self.W_x(X_t[:, frame, :, :, :]).chunk(4, dim = 1)
            W_hi, W_hf, W_hc, W_ho = self.W_h(H_t_1).chunk(4, dim = 1)
            W_ci, W_cf, W_co = self.W_c.chunk(3, dim = 0)
            b_i, b_f, b_c, b_o = self.b.chunk(4, dim = 0)
    
            i_t = self.sigmoid(W_xi + W_hi + W_ci * C_t_1 + b_i)
            f_t = self.sigmoid(W_xf + W_hf + W_cf * C_t_1 + b_f)
            C_t = f_t * C_t_1 + i_t * self.tanh(W_xc + W_hc + b_c)
            o_t = self.sigmoid(W_xo + W_ho + W_co * C_t + b_o)
            H_t = o_t * self.tanh(C_t)

            H_t = self.layer_norm(H_t)

            H_t_list.append(H_t)
            C_t_list.append(C_t)

            H_t_1 = H_t
            C_t_1 = C_t

        H_t_stack = torch.stack(H_t_list, dim = 1)
        C_t_stack = torch.stack(C_t_list, dim = 1)

        return H_t_stack, C_t_stack

class ConvLSTM(nn.Module):

    def __init__(self,
                 in_channels: Tuple[int],
                 out_channels: int,
                 embed_dim: Tuple[int],
                 patch_size: int,
                 layers: int):

        super().__init__()
        self.patch_size = patch_size
        
        self.lstm_layers = nn.ModuleList()

        for layer in range(layers):
            self.lstm_layers.append(ConvLSTMLayer(in_channels= in_channels[layer],
                                                  embed_dim= embed_dim[layer],
                                                  patch_size= patch_size))

        self.patch_embed = nn.Unfold(kernel_size=4,
                                     stride=4,
                                     padding=0)

        self.reconstruct = nn.Sequential(nn.Conv2d(in_channels=embed_dim[-1] + embed_dim[-2],
                                                   out_channels= out_channels * patch_size,
                                                   kernel_size=1,
                                                   stride=1),
                                         nn.Flatten(start_dim=2,
                                                    end_dim=3),
                                         nn.Fold(kernel_size= 4,
                                                 stride= 4,
                                                 output_size= (64, 64)))

    def forward(self,
                X: torch.tensor) -> torch.Tensor:
        
        B, T, _, _, _ = X.shape
        img_patch_list = []
        forecast_outputs = []
        img_reconstructed_list = []
        for i in range(T):
            X_i = self.patch_embed(X[:, i])
            img_patch_list.append(X_i)

        X_t = torch.stack(img_patch_list, dim= 1).reshape(B, T, -1, self.patch_size, self.patch_size)
        # print(X_t.shape)
        for layer in self.lstm_layers:
            X_t, _ = layer(X_t)
            forecast_outputs.append(X_t)

        forecast_concat = torch.cat(forecast_outputs[-2:], dim=2)
        for i in range(T):
            X_t_plus_1_single = self.reconstruct(forecast_concat[:, i])
            img_reconstructed_list.append(X_t_plus_1_single)

        X_t_plus_1 = torch.stack(img_reconstructed_list, dim= 1)
        
        return X_t_plus_1
