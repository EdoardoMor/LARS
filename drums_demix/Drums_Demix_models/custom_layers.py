import math
import torch
from torch import nn
from torch import Tensor
from unet import UNetEncoderBlock


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return x


class SelfAttnBlock(nn.Module):
    def __init__(self, n_blocks: int = 4, d_model: int = 16, n_head: int = 4, dim_feedforward: int = 256,
                 attn_mode: str = 'time'):
        super().__init__()

        self.attn_mode = attn_mode
        self.transformer_layers = nn.Sequential(
            PositionalEncoding(d_model=d_model),
        )

        for __ in range(n_blocks):
            self.transformer_layers.append(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward)
            )

    def _forward_time_attn(self, x):
        x_shape = x.size()  # (N x C x F x T)

        # (N x C x F x T) -> (NC x F x T) -> (T x NC x F)
        z = x.reshape(-1, x_shape[-2], x_shape[-1]).permute(2, 0, 1)

        z = self.transformer_layers(z)

        # (T x NC x F) -> (NC x F x T) -> (N x C x F x T)
        z = z.permute(1, 2, 0).reshape(x_shape)

        return z

    def _forward_channels_attn(self, x):
        x_shape = x.size()  # (N x C x F x T)

        # (N x C x F x T) -> (N x C x FT) ->  (C x N x FT)
        z = x.reshape(x_shape[0], x_shape[1], -1).permute(1, 0, 2)

        z = self.transformer_layers(z)

        # (C x N x FT) -> (N x C x FT) -> (N x C x F x T)
        z = z.permute(1, 0, 2).reshape(x_shape)

        return z

    def forward(self, x):
        if self.attn_mode == 'time':
            return self._forward_time_attn(x)
        elif self.attn_mode == 'channels':
            return self._forward_channels_attn(x)
        raise ValueError(f'Attention mode `{self.attn_mode}` not recognized.')


class AttnUnetEncoderBlock(UNetEncoderBlock):
    def __init__(self, in_channels: int, out_channels: int, d_model: int, dim_feedforward: int, n_head: int = 4):
        super().__init__(in_channels, out_channels)
        self.attn_skip = SelfAttnBlock(n_blocks=4, d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward,
                                       attn_mode='time')

    def forward(self, x):
        c = self.conv(x)
        c_attn = self.attn_skip(c)
        y = self.act(self.bn(c))
        return y, c_attn
