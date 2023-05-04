import torch
import soundfile as sf
# Load one tensor
tensor_model = torch.jit.load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/test2.pt")
tensor = list(tensor_model.parameters())[0]
sf.write('test2.wav', tensor.detach().cpu().numpy().T, 44100, subtype='PCM_16')