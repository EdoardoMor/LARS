import os
import sys
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import torchaudio as ta
from pathlib import Path
from getopt import getopt
from statistics import mean, StatisticsError
from separator import Separator
from utils import drumkit_names, stem_names, stem_composition

stem_name_mapping = {
    'kick': 'KD',
    'snare': 'SD',
    'toms': 'TT',
    'hihat': 'HH',
    'cymbals': 'CY'
}


def nsdr(y_pred, y_true, eps: float = 1e-7, reduction: str = 'mean'):
    assert reduction in ('none', 'mean', 'sum'), f'Reduction mode \"{reduction}\" not recognized.'

    signal = torch.square(y_true).mean(dim=[-2, -1])
    distortion = torch.square(y_true - y_pred).mean(dim=[-2, -1])

    if signal < np.finfo(np.float32).eps:
        signal = torch.zeros_like(signal)

    if distortion < np.finfo(np.float32).eps:
        distortion = torch.zeros_like(distortion)

    ratio = 10 * torch.log10((signal + eps) / (distortion + eps))

    if reduction == 'mean':
        return ratio.mean(0)
    elif reduction == 'sum':
        return ratio.sum(0)
    elif reduction == 'none':
        return ratio


def evaluate(csv: str, device: str = 'cpu', wiener_filter: bool = False, wiener_exponent: float = 1.0):
    if not torch.cuda.is_available():
        device = 'cpu'

    root = Path('StemGMD')
    results_dir = Path('eval_results')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    sources = pd.read_csv(csv)
    # sources = pd.DataFrame(
    #     [['audio/drummer1/eval_session/brooklyn', '10_soul-groove10_102_beat_4-4.wav'],
    #      ['audio/drummer1/eval_session/east_bay', '10_soul-groove10_102_beat_4-4.wav']]
    # )

    separator = Separator(wiener_filter=wiener_filter, wiener_exponent=wiener_exponent, device=device)

    sdr_present = {}
    sdr_non_present = {}

    for kit in drumkit_names:
        sdr_present[kit] = {s: [] for s in stem_names}
        sdr_non_present[kit] = {s: [] for s in stem_names}

    for row in sources.itertuples(index=False):
        idx, basepath, filename = row[:3]
        basepath = Path(basepath)

        pred = separator(root.joinpath(basepath, 'mixture', filename))

        for stem in stem_names:
            y_true, sr = ta.load(str(root.joinpath(basepath, stem, filename)))
            y_true = y_true.to(device)

            indicator = bool(sum([row[sources.columns.get_loc(inst)] for inst in stem_composition[stem]]))

            if indicator:
                sdr_present[basepath.stem][stem].append(
                    nsdr(pred[stem], y_true).item()
                )
            else:
                sdr_non_present[basepath.stem][stem].append(
                    nsdr(pred[stem], y_true).item()
                )

    with open(results_dir.joinpath(f'larsnet_nsdr_present_{Path(csv).stem}.pkl'), 'wb') as f:
        pickle.dump(sdr_present, f)

    with open(results_dir.joinpath(f'larsnet_nsdr_non_present_{Path(csv).stem}.pkl'), 'wb') as f:
        pickle.dump(sdr_non_present, f)

    print_results(sdr_present)
    print_results_latex(sdr_present)
    print_results_latex(sdr_non_present)


def print_single_result(metrics: dict or str):
    if isinstance(metrics, str):
        with open(Path('eval_results', metrics), 'rb') as f:
            metrics = pickle.load(f)

    all_all_ls = []
    for kit in drumkit_names:
        print(kit.capitalize())
        all_ls = []
        for stem, ls in metrics[kit].items():
            print(f'- SDR {stem}: {ls}')
            all_ls += ls
        print(f'- SDR All: {all_ls}')
        all_all_ls += all_ls
    print(f'\nSDR All-All: {all_all_ls}')


def print_results(metrics: dict or str):
    if isinstance(metrics, str):
        with open(Path('eval_results', metrics), 'rb') as f:
            metrics = pickle.load(f)

    all_all_ls = []
    for kit in drumkit_names:
        print(kit.capitalize())
        all_ls = []
        for stem, ls in metrics[kit].items():
            print(f'- SDR {stem}: {np.asarray(ls).mean()}')
            all_ls += ls
        print(f'- SDR All: {np.asarray(all_ls).mean()}')
        all_all_ls += all_ls
    print(f'\nSDR All-All: {np.asarray(all_all_ls).mean()}')


def print_results_latex(metrics: dict or str):
    if isinstance(metrics, str):
        with open(Path('eval_results', metrics), 'rb') as f:
            metrics = pickle.load(f)

    print('\n\n')

    num_strings = ['$400$ ($100\\%$)', '$400$ ($100\\%$)', '$40$ ($10\\%$)', '$390$ ($97.5\\%$)', '$50$ ($12.5\\%$)']

    for stem, num in zip(stem_names, num_strings):
        last_col = []
        row = f'& {stem_name_mapping[stem]} & {num}'
        for kit in drumkit_names:
            row += f' & ${np.asarray(metrics[kit][stem]).mean():.2f}$'
            last_col += metrics[kit][stem]
        row += f' & ${np.asarray(last_col).mean():.2f}$'
        row += '\\\\'
        print(row)

    print('\\cmidrule(c){2-14}')

    row = '& Avg &'
    all_kits_all_stems = []
    for kit in drumkit_names:
        all_stems_one_kit = []
        for stem in stem_names:
            all_stems_one_kit += metrics[kit][stem]
            all_kits_all_stems += metrics[kit][stem]
        row += f' & ${np.asarray(all_stems_one_kit).mean():.2f}$'
    row += f' & ${np.asarray(all_kits_all_stems).mean():.2f}$'
    row += '\\\\'
    print(row)


def main(argv):
    device = 'cpu'
    opts, args = getopt(argv, "d:w:", ["device=", "wiener="])

    wiener_filter = False
    wiener_exponent = 1.0

    for opt, arg in opts:
        if opt in ('-d', '--device'):
            if arg[:4] != 'cuda' and arg != 'cpu':
                raise ValueError(f'Device \"{arg}\" not recognized.')
            if arg[:4] == 'cuda' and not torch.cuda.is_available():
                warnings.warn("Cuda is not available, using CPU instead.")
            else:
                device = arg
        if opt in ('-w', '--wiener'):
            if float(arg) > 0:
                wiener_filter = True
                wiener_exponent = float(arg)

    if wiener_filter:
        print(f'> Applying Wiener filter with α={wiener_exponent}')
    else:
        print('No Wiener filter')

    evaluate(csv='csv/test_eval_session_presence.csv', device=device,
             wiener_filter=wiener_filter, wiener_exponent=wiener_exponent)


if __name__ == '__main__':
    main(sys.argv[1:])
