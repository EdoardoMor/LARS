import torchaudio as ta
from pathlib import Path

train_drumkit_names = (
    'brooklyn',
    'east_bay',
    'heavy',
    'portland',
    'retro_rock',
    'socal'
)

unseen_drumkit_names = (
    'bluebird',
    'detroit_garage',
    'motown_revisited',
    'roots'
)

drumkit_names = (
    'brooklyn',
    'east_bay',
    'heavy',
    'portland',
    'retro_rock',
    'socal',
    'bluebird',
    'detroit_garage',
    'motown_revisited',
    'roots'
)

stem_names = (
    'kick',
    'snare',
    'toms',
    'hihat',
    'cymbals'
)

instrument_names = (
    'kick',
    'snare',
    'hi_tom',
    'mid_tom',
    'low_tom',
    'hihat_open',
    'hihat_closed',
    'crash_left',
    'ride'
)

stem_composition = {
    'kick': ['kick'],
    'snare': ['snare'],
    'toms': ['hi_tom', 'mid_tom', 'low_tom'],
    'hihat': ['hihat_closed', 'hihat_open'],
    'cymbals': ['ride', 'crash_left']
}


def load_audio_file(path: str or Path, frame_offset: int, num_frames: int, force_stereo: bool = True):
    wav, sr = ta.load(str(path), frame_offset=frame_offset, num_frames=num_frames)
    if force_stereo and wav.size(0) == 1:
        return wav.repeat(2, 1)
    return wav, sr
