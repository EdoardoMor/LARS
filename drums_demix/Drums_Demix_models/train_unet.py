import os
import sys
import yaml
import torch
import shutil
import random
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
from getopt import getopt
from torch import nn, optim
from dataset import StemGMDDataset
from unet import UNet, UNetW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Any
from evaluation import nsdr
from utils import stem_names


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_epoch(model: nn.Module, loader: DataLoader, writer: SummaryWriter, epoch: int, criterion: Callable,
                optimizer: optim.Optimizer, scheduler: Any, device: str, mode: str):
    model.train()

    running_loss = 0.
    n_iter = (epoch - 1) * len(loader)

    pbar = tqdm(loader)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        if mode == 'stft':
            x, __ = model.utils.batch_stft(x)
            y, __ = model.utils.batch_stft(y)

        y_hat, __ = model(x)

        if mode == 'stft':
            y = model.utils.trim_freq_dim(y)
            y_hat = model.utils.trim_freq_dim(y_hat)

        loss = criterion(y_hat, y)
        loss.backward()

        optimizer.step()

        if scheduler is not None:
            # Possibly add some condition...
            scheduler.step()

        pbar.set_description(f'Train loss: {loss.item():e}')
        writer.add_scalar('Loss/train_batch', loss.item(), n_iter)
        running_loss += loss.item() * x.size(0)
        n_iter += 1

    epoch_loss = running_loss / len(loader.dataset)
    writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)

    return epoch_loss


def val_epoch(model: nn.Module, loader: DataLoader, writer: SummaryWriter, epoch: int, criterion: Callable,
              device: str, mode: str):
    model.eval()

    running_loss = 0.
    running_sdr = 0.

    with torch.no_grad():
        pbar = tqdm(loader)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if mode == 'stft':
                X, X_phi = model.utils.batch_stft(x)
                Y, __ = model.utils.batch_stft(y)

                Y_hat, __ = model(X)

                loss = criterion(model.utils.trim_freq_dim(Y_hat),
                                 model.utils.trim_freq_dim(Y))

                y_hat = model.utils.batch_istft(Y_hat, X_phi, trim_length=x.size(-1))

            else:
                y_hat, __ = model(x)
                loss = criterion(y_hat, y)

            sdr = nsdr(y_hat, y)

            pbar.set_description(f'Val loss: {loss.item():e} - nSDR: {sdr.item():e}')

            running_loss += loss.item() * x.size(0)
            running_sdr += sdr.item() * x.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_sdr = running_sdr / len(loader.dataset)
    writer.add_scalar('Loss/val_epoch', epoch_loss, epoch)
    writer.add_scalar('nSDR/val_epoch', epoch_sdr, epoch)

    return epoch_loss, epoch_sdr


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, train_loss: float, val_loss: float,
                    val_sdr: float, path: str or Path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_sdr': val_sdr

    }, path
    )


def train(stem: str, config: dict, device: str, resume: str or bool = False):
    sample_rate = int(config['global']['sample_rate'])
    segment = float(config['global']['segment'])
    shift = float(config['global']['shift'])
    n_workers = int(config['global']['n_workers'])
    prefetch_factor = int(config['global']['prefetch_factor'])

    augmentation_prob = float(config['data_augmentation']['augmentation_prob'])
    kit_swap_augment_prob = float(config['data_augmentation']['kit_swap_augment_prob'])
    doubling_augment_prob = float(config['data_augmentation']['doubling_augment_prob'])
    channel_swap_augment_prob = float(config['data_augmentation']['channel_swap_augment_prob'])
    pitch_shift_augment_prob = float(config['data_augmentation']['pitch_shift_augment_prob'])
    saturation_augment_prob = float(config['data_augmentation']['saturation_augment_prob'])
    remix_augment_prob = float(config['data_augmentation']['remix_augment_prob'])

    training_mode = str(config[stem]['training_mode'])
    epochs = int(config[stem]['epochs'])
    batch_size = int(config[stem]['batch_size'])
    learning_rate = float(config[stem]['learning_rate'])

    F = int(config[stem]['F'])
    T = int(config[stem]['T'])

    if resume:
        model_id = resume.parts[-2]
        saved_model_dir = resume.parent
        log_dir = Path('runs', *resume.parts[-3:-1])
    else:
        model_id = f'unet__v3__mode={training_mode}__freqBN__size=(2,{F},{T})__batch={batch_size}__lr={learning_rate}__segment={segment}__shift={shift}'
        saved_model_dir = Path('saved_models', stem, model_id)
        log_dir = Path('runs', stem, model_id)

    print(f'Torch device: \"{device}\"')
    print(f'Target stem: \"{stem}\"')
    print(f'Input size: {(2, F, T)}')
    print(f'Training mode: \"{training_mode}\"')
    print(f'Model ID: \"{model_id}\"\n')

    if training_mode == 'stft':
        model = UNet(input_size=(2, F, T), device=device)

    elif training_mode == 'w2w':
        model = UNetW(input_size=(2, F, T), device=device)

    else:
        raise ValueError(f'Training mode `{training_mode}` not recognized.')

    train_dataset = StemGMDDataset(root='StemGMD',
                                   stem=stem,
                                   sources='csv/train.csv',
                                   segment=segment,
                                   shift=shift,
                                   sample_rate=sample_rate,
                                   augmentation_prob=augmentation_prob,
                                   kit_swap_augment_prob=kit_swap_augment_prob,
                                   doubling_augment_prob=doubling_augment_prob,
                                   channel_swap_augment_prob=channel_swap_augment_prob,
                                   pitch_shift_augment_prob=pitch_shift_augment_prob,
                                   saturation_augment_prob=saturation_augment_prob,
                                   remix_augment_prob=remix_augment_prob
                                   )

    val_dataset = StemGMDDataset(root='StemGMD',
                                 stem=stem,
                                 sources='csv/validation.csv',
                                 segment=segment,
                                 shift=shift,
                                 sample_rate=sample_rate,
                                 )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, prefetch_factor=prefetch_factor)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, prefetch_factor=prefetch_factor)

    writer = SummaryWriter(log_dir=log_dir)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    scheduler = None
    start_epoch = 0

    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'\nResuming from epoch {start_epoch}...')

    for epoch in range(epochs):
        epoch += start_epoch + 1

        print(f'Epoch: {epoch}/{epochs + start_epoch}')

        train_loss = train_epoch(model, train_loader, writer, epoch, criterion, optimizer, scheduler, device,
                                 mode=training_mode)
        val_loss, val_sdr = val_epoch(model, val_loader, writer, epoch, criterion, device, mode=training_mode)

        print(f'Train loss: {train_loss} - Val loss: {val_loss}\n')

        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)

        ckpt_path = saved_model_dir.joinpath(f'ckpt_epoch_{epoch:03d}.pth')
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_sdr, ckpt_path)
        shutil.copyfile(ckpt_path, saved_model_dir.joinpath('last.pth'))

    writer.close()


def main(argv):
    seed_everything(42)

    opts, args = getopt(argv, "s:d:", ["stem=", "device="])

    stem = None
    device = 'cpu'

    for opt, arg in opts:
        if opt in ('-s', '--stem'):
            if arg not in stem_names:
                raise ValueError(f'Stem \"{arg}\" not recognized. Only {stem_names} are allowed')
            stem = arg
        elif opt in ('-d', '--device'):
            if arg[:4] != 'cuda' and arg != 'cpu':
                raise ValueError(f'Device \"{arg}\" not recognized.')
            if arg[:4] == 'cuda' and not torch.cuda.is_available():
                warnings.warn("Cuda is not available, using CPU instead.")
            else:
                device = arg

    if stem is None:
        raise ValueError('Target stem must be specified via \"-s\" or \"--stem\"')

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    resume = False

    if resume:
        resume = Path(str(resume))
        if not os.path.exists(str(resume)):
            raise RuntimeError(f'Could not find model at {resume}')
        if resume.parts[-3] != stem:
            raise ValueError(f'Trying to resume a \"{resume.parts[-3]}\" model but \"{stem}\" was requested!')

    train(stem=stem, config=config, device=device, resume=resume)


if __name__ == '__main__':
    main(sys.argv[1:])
