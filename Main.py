import numpy as np
import torch

from DataLoader import load_data, save_sound
from AmpAutoencoder import Autoencoder, UNet
from Spectogram import real_wavelet_transform, inverse_wavelet_transform

source_data, target_data = load_data()
#source_data = np.load('source_spectogram.npy')#real_wavelet_transform(source_data_or,fs=44100,freq_range=(20,20000),freq_subdiv=50,stride=2)
#target_data = np.load('target_spectogram.npy')#real_wavelet_transform(target_data_or,fs=44100,freq_range=(20,20000),freq_subdiv=50,stride=2)
#np.save('source_spectogram.npy',source_data)
#np.save('target_spectogram.npy',target_data)

def get_random_snippet_batch(X,Y,batch_size,num_samples_per_snippet):
	input_batch = np.zeros((batch_size,1,num_samples_per_snippet))
	output_batch = np.zeros((batch_size,1,num_samples_per_snippet))
	for i in range(batch_size):
		
		start = np.random.randint(len(X)-num_samples_per_snippet)
		
		input_batch[i,0,:] = X[start:start+num_samples_per_snippet]
		output_batch[i,0,:] = Y[start:start+num_samples_per_snippet]
			

	return input_batch, output_batch
	
	
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_iterations = 5000
learning_rate = 1e-3
batch_size = 50
num_samples_per_snippet = 9600


model = UNet()
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(num_iterations):
	input, target = get_random_snippet_batch(source_data,target_data,batch_size,num_samples_per_snippet)
		
	input = torch.as_tensor(input.astype(np.float32)).to(device)
	target = torch.as_tensor(target.astype(np.float32)).to(device)
	
	input_batch = torch.autograd.Variable(input)
	output_target = torch.autograd.Variable(target)
	
	output = model.forward(input_batch)
	
	loss = criterion(10*output[:,:,100:-100],10*output_target[:,:,100:-100])

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('iteration [{}/{}], loss:{:.4f}'.format(i,num_iterations, loss.data.item()))
	
torch.save(model.state_dict(),'model_save.pt')
output_target = np.zeros_like(source_data)
n = len(source_data)//96000
print(len(source_data)%96000)
for i in range(n):
	input = source_data[i*96000:(i+1)*96000][np.newaxis,np.newaxis,:]
	input = torch.as_tensor(input.astype(np.float32)).to(device)
	output_target[i*96000:(i+1)*96000] = model.forward(input).detach().cpu().numpy()[0,0,:]


save_sound(output_target, 'target_repl', rate=44100)

import matplotlib.pyplot as plt
plt.plot(source_data[4400:8800])
plt.plot(target_data[4400:8800])
plt.plot(output_target[4400:8800])

plt.show()
