import math
import torch
import random
import pandas as pd
import torchaudio as ta
from typing import List
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn import functional as F
from data_augmentation import Compose, RandomApply, Gain, PitchShift, Saturation, ChannelSwap, Doubling
from utils import load_audio_file, stem_names, train_drumkit_names


class StemGMDDataset(Dataset):

    def __init__(self, *,
                 root: str or Path,
                 stem: str,
                 sources: str or Path,      # csv file path
                 segment: float = None,     # seconds
                 shift: float = None,       # seconds
                 sample_rate: int = 44100,
                 augmentation_prob=0.,
                 kit_swap_augment_prob=0.,
                 doubling_augment_prob=0.,
                 channel_swap_augment_prob=0.,
                 pitch_shift_augment_prob=0.,
                 saturation_augment_prob=0.,
                 remix_augment_prob=0.,
                 ):

        self.stem_names = list(stem_names)
        self.drumkit_names = list(train_drumkit_names)

        self.root = Path(root)
        self.stem = stem
        self.sources = pd.read_csv(sources)
        self.segment = segment
        self.shift = shift
        self.sample_rate = sample_rate

        self.augmentation_prob = augmentation_prob
        self.kit_swap_augment_prob = kit_swap_augment_prob
        self.doubling_augment_prob = doubling_augment_prob
        self.channel_swap_augment_prob = channel_swap_augment_prob
        self.pitch_shift_augment_prob = pitch_shift_augment_prob
        self.saturation_augment_prob = saturation_augment_prob
        self.remix_augment_prob = remix_augment_prob

        self.num_examples = self._compute_num_examples()

    def _compute_num_examples(self) -> List:
        if self.segment is None:
            return [len(self.sources)]

        print('Computing number of examples...')
        num_examples = []
        segment_len = self.segment * self.sample_rate
        shift_len = self.shift * self.sample_rate
        for row in self.sources.itertuples(index=False):
            basepath, filename = row[:2]
            info = ta.info(self.root.joinpath(basepath, 'mixture', filename))
            track_duration = info.num_frames
            if self.segment is None or track_duration < segment_len:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - segment_len) / shift_len) + 1)
            num_examples.append(examples)
        print(f'The dataset contains {len(num_examples)} files for a total of {sum(num_examples)} pairs of examples.\n')
        return num_examples

    @staticmethod
    def _fix_length(segment: torch.Tensor, num_frames: int):
        x = segment[..., :num_frames]
        x = F.pad(x, (0, num_frames - x.size(-1)))
        return x

    def _data_augmentation(self, basepath: str or Path, filename: str, frame_offset: int, num_frames: int):
        with torch.no_grad():
            basepath = Path(basepath)

            do_kit_swap = RandomApply.bernoulli_trial(p=self.kit_swap_augment_prob)
            do_remix = RandomApply.bernoulli_trial(p=self.remix_augment_prob)

            duration = int(math.ceil(self.sample_rate * self.segment))

            composed_augmentation = Compose([
                RandomApply([PitchShift(duration, sample_rate=44100, pitch_shift_min=-3.0, pitch_shift_max=3.0)],
                            p=self.pitch_shift_augment_prob),
                RandomApply([Saturation(min_beta=1, max_beta=5)],
                            p=self.saturation_augment_prob),
                RandomApply([ChannelSwap()],
                            p=self.channel_swap_augment_prob),
                RandomApply([Gain(min_gain=-12.0, max_gain=0.)],
                            p=float(do_remix))
            ])

            stems = torch.zeros(len(self.stem_names), 2, duration)

            for i, stem_name in enumerate(self.stem_names):

                if do_kit_swap and stem_name != self.stem:
                    kit = random.choice(self.drumkit_names)
                else:
                    kit = basepath.stem

                wav_path = self.root.joinpath(basepath.parent, kit, stem_name, filename)
                wav, __ = load_audio_file(wav_path, frame_offset, num_frames)
                wav = F.pad(wav[..., :duration], (0, duration - wav.size(-1), 0, 0))

                wav = RandomApply([Doubling(stem_name, self.root, basepath, filename, frame_offset, num_frames,
                                            drumkits=[k for k in self.drumkit_names if k != kit])],
                                  p=self.doubling_augment_prob)(wav)

                stems[i] = composed_augmentation(wav)

            idx = self.stem_names.index(self.stem)
            mix = stems.sum(0)
            stem = stems[idx]

            return mix, stem

    def _getitem(self, index):
        for num, src in zip(self.num_examples, self.sources.iloc):
            if index >= num:
                index -= num
                continue
            basepath, filename = src
            break

        frame_offset = 0
        num_frames = -1

        if self.segment is not None:
            frame_offset = int(self.sample_rate * self.shift * index)
            num_frames = int(math.ceil(self.sample_rate * self.segment))

        if RandomApply.bernoulli_trial(p=self.augmentation_prob):
            mix, stem = self._data_augmentation(basepath, filename, frame_offset, num_frames)
        else:
            mix, _ = load_audio_file(self.root.joinpath(basepath, 'mixture', filename), frame_offset, num_frames)
            stem, _ = load_audio_file(self.root.joinpath(basepath, self.stem, filename), frame_offset, num_frames)

        if self.segment:
            mix = self._fix_length(mix, num_frames)
            stem = self._fix_length(stem, num_frames)

        return mix, stem

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        return self._getitem(index)
