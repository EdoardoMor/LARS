import torch
from unet import UNet
from custom_layers import AttnUnetEncoderBlock
from typing import Tuple


class AttnUNet(UNet):

    def __init__(self, input_size: Tuple[int, ...] = (2, 1024, 512), power: float = 1.0, device: str or None = None):
        super().__init__(input_size=input_size, power=power, device=device)

        # Encoder
        self.enc1 = AttnUnetEncoderBlock(input_size[0], 16, d_model=1024, dim_feedforward=1024, n_head=4)
        self.enc2 = AttnUnetEncoderBlock(16, 32, d_model=512, dim_feedforward=512, n_head=4)
        self.enc3 = AttnUnetEncoderBlock(32, 64, d_model=256, dim_feedforward=256, n_head=4)
        self.enc4 = AttnUnetEncoderBlock(64, 128, d_model=128, dim_feedforward=128, n_head=4)
        self.enc5 = AttnUnetEncoderBlock(128, 256, d_model=64, dim_feedforward=64, n_head=4)
        self.enc6 = AttnUnetEncoderBlock(256, 512, d_model=32, dim_feedforward=32, n_head=4)

        if device is not None:
            self.to(device)


class AttnUNetW(AttnUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = torch.as_tensor(x)
        if x.dim() == 1:
            x = x.repeat(2, 1)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mag, phase = self.utils.batch_stft(x)
        mag_hat, mask = super().forward(mag)
        x_hat = self.utils.batch_istft(mag_hat, phase, trim_length=x.size(-1))
        return x_hat, mask
