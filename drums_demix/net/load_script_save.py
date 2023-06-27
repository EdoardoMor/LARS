import os
import sys
import yaml
import torch
import shutil
import random
import warnings
import numpy as np
import torchaudio as ta
from tqdm import tqdm
from pathlib import Path
from getopt import getopt
from torch import nn, optim
from unet import UNet, UNetW, UNetUtils

F = 2048
T = 512

model = UNet(input_size=(2, F, T), device="cpu")

checkpoint_kick = torch.load('C:\\POLIMI\\MAE_Capstone\\DrumsDemix\\drums_demix\\Drums_Demix_models\\saved_models\\kick\\unet__v3__mode=stft__freqBN__size=(2,2048,512)__batch=24__lr=0.0001__segment=11.85__shift=2.0\\ckpt_epoch_021.pth', map_location=torch.device('cpu'))
checkpoint_snare = torch.load('C:\\POLIMI\\MAE_Capstone\\DrumsDemix\\drums_demix\\Drums_Demix_models\\saved_models\\snare\\unet__v3__mode=stft__freqBN__size=(2,2048,512)__batch=24__lr=0.0001__segment=11.85__shift=2.0\\ckpt_epoch_022.pth', map_location=torch.device('cpu'))
checkpoint_toms = torch.load('C:\\POLIMI\\MAE_Capstone\\DrumsDemix\\drums_demix\\Drums_Demix_models\\saved_models\\toms\\unet__v3__mode=stft__freqBN__size=(2,2048,512)__batch=24__lr=0.0001__segment=11.85__shift=2.0\\ckpt_epoch_020.pth', map_location=torch.device('cpu'))
checkpoint_hihat = torch.load('C:\\POLIMI\\MAE_Capstone\\DrumsDemix\\drums_demix\\Drums_Demix_models\\saved_models\\hihat\\unet__v3__mode=stft__freqBN__size=(2,2048,512)__batch=24__lr=0.0001__segment=11.85__shift=2.0\\ckpt_epoch_022.pth', map_location=torch.device('cpu'))
checkpoint_cymbals = torch.load('C:\\POLIMI\\MAE_Capstone\\DrumsDemix\\drums_demix\\Drums_Demix_models\\saved_models\\cymbals\\unet__v3__mode=stft__freqBN__size=(2,2048,512)__batch=24__lr=0.0001__segment=11.85__shift=2.0\\ckpt_epoch_020.pth', map_location=torch.device('cpu'))

state_dict_kick = checkpoint_kick['model_state_dict']
state_dict_snare = checkpoint_snare['model_state_dict']
state_dict_toms = checkpoint_toms['model_state_dict']
state_dict_hihat = checkpoint_hihat['model_state_dict']
state_dict_cymbals = checkpoint_cymbals['model_state_dict']

dict_list = [state_dict_kick, state_dict_snare, state_dict_toms, state_dict_hihat, state_dict_cymbals]
dict_name_list = ["my_scripted_module_kick.pt", "my_scripted_module_snare.pt", "my_scripted_module_toms.pt", "my_scripted_module_hihat.pt", "my_scripted_module_cymbals.pt"]

for dict, name in zip(dict_list, dict_name_list):

    print(name)

    model.load_state_dict(dict)
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save(name)


print("test")










'''IN CASE OF TRACE

path = Path('C:/POLIMI/MAE_Capstone/audio/1_funk-groove1_138_beat_4-4_socal.wav')

x, sr = ta.load(str(path))

if x.dim() == 1:
    x = x.repeat(2, 1)
if x.dim() == 2:
    x = x.unsqueeze(0)

x = x.to("cpu")
utils = UNetUtils(device="cpu")
mag, phase = utils.batch_stft(x)

scripted_model = torch.jit.trace(model, mag)
scripted_model.save('my_scripted_module.pt')

'''