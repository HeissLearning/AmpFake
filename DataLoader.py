import numpy as np
from scipy.io import wavfile
import torch

source_file = 'Data/X.pt'
target_file = 'Data/Y.pt'

rate = 44100

def load_data():
	source_data = torch.load(source_file).numpy()
	target_data = torch.load(target_file).numpy()
	
	return source_data.reshape(-1), target_data.reshape(-1)
	
def save_sound(data,filename,rate=44100):
	wavfile.write(filename+'.wav',rate,data)
