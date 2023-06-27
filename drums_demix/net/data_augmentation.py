import torch
import random
import augment
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T
from pathlib import Path
from utils import load_audio_file


class Doubling(nn.Module):
    def __init__(self, stem: str, root: Path, basepath: Path, filename: str, frame_offset: int, num_frames: int,
                 drumkits: list):
        super().__init__()
        self.stem = stem
        self.root = root
        self.basepath = basepath
        self.filename = filename
        self.frame_offset = frame_offset
        self.num_frames = num_frames
        self.drumkits = drumkits

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        kit = random.choice(self.drumkits)
        path = self.root.joinpath(self.basepath.parent, kit, self.stem, self.filename)
        double, __ = load_audio_file(path, self.frame_offset, self.num_frames)
        double = F.pad(double[..., :audio.size(-1)], (0, audio.size(-1) - double.size(-1), 0, 0))
        return 0.5 * (audio + double)


class ChannelSwap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio.flip(0)


class Saturation(nn.Module):
    def __init__(self, min_beta: float = 1, max_beta: float = 5):
        super().__init__()
        self.min_beta = min_beta
        self.max_beta = max_beta

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        beta = random.uniform(self.min_beta, self.max_beta)
        audio = torch.tanh(beta * audio)
        return audio


class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = T.Vol(gain, gain_type="db")(audio)
        return audio


class PitchShift(nn.Module):
    def __init__(self, n_samples, sample_rate, pitch_shift_min=-3.0, pitch_shift_max=3.0):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.pitch_shift_cents_min = int(pitch_shift_min * 100)
        self.pitch_shift_cents_max = int(pitch_shift_max * 100)
        self.src_info = {"rate": self.sample_rate}

    def _process(self, x):
        n_steps = random.randint(self.pitch_shift_cents_min, self.pitch_shift_cents_max)
        effect_chain = augment.EffectChain().pitch("-q", n_steps).rate(self.sample_rate)
        num_channels = x.shape[0]
        target_info = {
            "channels": num_channels,
            "length": self.n_samples,
            "rate": self.sample_rate,
        }

        y = effect_chain.apply(x, src_info=self.src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()

        if y.shape[1] != x.shape[1]:
            if y.shape[1] > x.shape[1]:
                y = y[:, : x.shape[1]]
            else:
                y0 = torch.zeros(num_channels, x.shape[1]).to(y.device)
                y0[:, : y.shape[1]] = y
                y = y0
        return y

    def forward(self, audio):
        if audio.ndim == 3:
            for b in range(audio.shape[0]):
                audio[b] = self._process(audio[b])
            return audio
        else:
            return self._process(audio)


class RandomApply(nn.Module):
    """Apply randomly a list of transformations with a given probability.
    Args:
        transforms (list or tuple or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    @staticmethod
    def bernoulli_trial(p: float):
        return torch.rand(1) < p

    def forward(self, x):
        if self.bernoulli_trial(self.p):
            for t in self.transforms:
                x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return


class Compose:
    """Data augmentation module that transforms any given data example with a chain of audio augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
