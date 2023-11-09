import yaml
import torch
import torchaudio as ta
from torch import nn
from tqdm import tqdm
from pathlib import Path
from unet import UNet, UNetW, UNetUtils


class Separator(nn.Module):

    def __init__(self, wiener_filter: bool = False, wiener_exponent: float = 1.0, device: str = 'cpu',
                 return_stft: bool = False):
        super().__init__()

        self.device = device
        self.wiener_filter = wiener_filter
        self.wiener_exponent = wiener_exponent
        self.return_stft = return_stft

        if wiener_filter:
            print(f'> Applying Wiener filter with α={self.wiener_exponent}')

        with open("saved_models.yaml", "r") as f:
            model_path = yaml.safe_load(f)

        F = 2048
        T = 512

        self.models = {}
        self.stems = model_path.keys()
        self.utils = UNetUtils(device=self.device)

        print('Load models...')
        pbar = tqdm(self.stems)
        for stem in pbar:
            checkpoint_path = Path(model_path[stem])
            pbar.set_description(f'{stem} {checkpoint_path.stem}')
            if self.wiener_filter or self.return_stft:
                model = UNet(input_size=(2, F, T), device=self.device)
            else:
                model = UNetW(input_size=(2, F, T), device=self.device)
            checkpoint = torch.load(str(checkpoint_path), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models[stem] = model

    @staticmethod
    def fix_dim(x):
        if x.dim() == 1:
            x = x.repeat(2, 1)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def separate(self, x):
        with torch.no_grad():
            out = {}
            x = x.to(self.device)

            print('Separate drums...')
            pbar = tqdm(self.models.items())
            for stem, model in pbar:
                pbar.set_description(stem)
                y, __ = model(x)
                out[stem] = y.squeeze(0).detach()

        return out

    def separate_wiener(self, x):
        with torch.no_grad():
            out = {}
            mag_pred = []

            x = self.fix_dim(x).to(self.device)
            mag, phase = self.utils.batch_stft(x)

            print('Separate drums...')
            pbar = tqdm(self.models.items())
            for stem, model in pbar:
                pbar.set_description(stem)
                __, mask = model(mag)
                mag_pred.append(
                    (mask * mag) ** self.wiener_exponent
                )

            pred_sum = sum(mag_pred)

            for stem, pred in zip(self.stems, mag_pred):
                wiener_mask = pred / (pred_sum + 1e-7)
                y = self.utils.batch_istft(mag * wiener_mask, phase, trim_length=x.size(-1))
                out[stem] = y.squeeze(0).detach()

        return out

    def separate_stft(self, x):
        with torch.no_grad():
            out = {}

            x = self.fix_dim(x).to(self.device)
            mag, phase = self.utils.batch_stft(x)

            print('Separate drum magnitude...')
            pbar = tqdm(self.models.items())
            for stem, model in pbar:
                pbar.set_description(stem)
                mag, __ = model(mag)
                out[stem] = torch.polar(mag, phase).squeeze(0).detach()

        return out

    def forward(self, x):
        if isinstance(x, str) or isinstance(x, Path):
            x, sr = ta.load(str(x))

        if self.return_stft:
            return self.separate_stft(x)
        elif self.wiener_filter:
            return self.separate_wiener(x)
        else:
            return self.separate(x)
