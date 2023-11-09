import torch
from pathlib import Path
from unet import UNet


F = 2048
T = 512

model = UNet(input_size=(2, F, T), device='cpu')

root = Path('checkpoints')

checkpoint_kick = torch.load(root.joinpath('kick.pth'), map_location=torch.device('cpu'))
checkpoint_snare = torch.load(root.joinpath('snare.pth'), map_location=torch.device('cpu'))
checkpoint_toms = torch.load(root.joinpath('toms.pth'), map_location=torch.device('cpu'))
checkpoint_hihat = torch.load(root.joinpath('hihat.pth'), map_location=torch.device('cpu'))
checkpoint_cymbals = torch.load(root.joinpath('cymbals.pth'), map_location=torch.device('cpu'))

state_dict_kick = checkpoint_kick['model_state_dict']
state_dict_snare = checkpoint_snare['model_state_dict']
state_dict_toms = checkpoint_toms['model_state_dict']
state_dict_hihat = checkpoint_hihat['model_state_dict']
state_dict_cymbals = checkpoint_cymbals['model_state_dict']

state_dict_list = [
    state_dict_kick,
    state_dict_snare,
    state_dict_toms,
    state_dict_hihat,
    state_dict_cymbals
]

scripted_model_names = [
    "my_scripted_module_kick.pt",
    "my_scripted_module_snare.pt",
    "my_scripted_module_toms.pt",
    "my_scripted_module_hihat.pt",
    "my_scripted_module_cymbals.pt"
]

for state_dict, name in zip(state_dict_list, scripted_model_names):
    model.load_state_dict(state_dict)
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(name)
