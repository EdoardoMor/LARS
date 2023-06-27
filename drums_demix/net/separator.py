import yaml
import torch
import soundfile as sf
import torchaudio as ta
from torch import nn
from tqdm import tqdm
from pathlib import Path
from unet import UNet, UNetW, UNetUtils


class Separator(nn.Module):

    def __init__(self, wiener_filter: bool = False, wiener_exponent: float = 1.0, device: str = 'cpu'):
        super().__init__()

        self.device = device
        self.wiener_filter = wiener_filter
        self.wiener_exponent = wiener_exponent

        if wiener_filter:
            print(f'> Applying Wiener filter with α={self.wiener_exponent}')

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        with open("inference_models.yaml", "r") as f:
            model_path = yaml.safe_load(f)

        self.models = {}
        self.stems = model_path.keys()
        self.utils = UNetUtils(device=self.device)

        print('Load model...')
        pbar = tqdm(self.stems)
        for stem in pbar:
            checkpoint_path = Path(model_path[stem])
            pbar.set_description(f'{stem} {checkpoint_path.stem}')
            if self.wiener_filter:
                model = UNet(input_size=(2, config[stem]['F'], config[stem]['T']), device=self.device)
            else:
                model = UNetW(input_size=(2, config[stem]['F'], config[stem]['T']), device=self.device)
            checkpoint = torch.load(str(checkpoint_path), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models[stem] = model

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

            if x.dim() == 1:
                x = x.repeat(2, 1)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            x = x.to(self.device)
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

    def forward(self, x):
        if isinstance(x, str) or isinstance(x, Path):
            x, sr = ta.load(str(x))
        if self.wiener_filter:
            return self.separate_wiener(x)
        else:
            return self.separate(x)


if __name__ == '__main__':
    wav_paths = [
        'C:/POLIMI/MAE_Capstone/audio/1_funk-groove1_138_beat_4-4_socal.wav',
    ]

    wav_paths = [Path(p) for p in wav_paths]

    separator = Separator(wiener_filter=True, device='cpu')

    for wav_path in wav_paths:
        pred = separator(wav_path)

        mix, __ = ta.load(str(wav_path))

        sf.write(f'{wav_path.stem}_{wav_path.parts[-3]}_MIX.wav', mix.cpu().numpy().T, 44100, subtype='PCM_16')

        for stem_name, wav in pred.items():
            sf.write(f'{wav_path.stem}_{wav_path.parts[-3]}_{stem_name.upper()}.wav', wav.cpu().numpy().T, 44100,
                     subtype='PCM_16')
