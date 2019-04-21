import numpy as np
import torch

class UNet(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		module_list = []
		module_list.append(torch.nn.Conv1d(1,32,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.MaxPool1d(2,2))
		module_list.append(torch.nn.BatchNorm1d(32))
		self.block1 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(32,64,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.MaxPool1d(2,2))
		module_list.append(torch.nn.BatchNorm1d(64))
		self.block2 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(64,128,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.MaxPool1d(2,2))
		module_list.append(torch.nn.BatchNorm1d(128))
		self.block3 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(128,256,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.MaxPool1d(2,2))
		module_list.append(torch.nn.BatchNorm1d(256))
		self.block4 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(256,512,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.MaxPool1d(2,2))
		module_list.append(torch.nn.BatchNorm1d(512))
		self.block5 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(512,1024,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.MaxPool1d(2,2))
		module_list.append(torch.nn.BatchNorm1d(1024))
		module_list.append(torch.nn.Conv1d(1024,1024,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.ConvTranspose1d(1024,512,20,stride=2,padding=9))
		module_list.append(torch.nn.BatchNorm1d(512))
		self.block6 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(1024,512,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(512))
		module_list.append(torch.nn.ConvTranspose1d(512,256,20,stride=2,padding=9))
		module_list.append(torch.nn.ReLU())
		self.blockU5 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(512,256,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(256))
		module_list.append(torch.nn.ConvTranspose1d(256,128,20,stride=2,padding=9))
		module_list.append(torch.nn.ReLU())
		self.blockU4 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(256,128,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(128))
		module_list.append(torch.nn.Conv1d(128,128,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(128))
		module_list.append(torch.nn.ConvTranspose1d(128,64,20,stride=2,padding=9))
		module_list.append(torch.nn.ReLU())
		self.blockU3 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(128,64,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(64))
		module_list.append(torch.nn.Conv1d(64,64,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(64))
		module_list.append(torch.nn.ConvTranspose1d(64,32,20,stride=2,padding=9))
		module_list.append(torch.nn.ReLU())
		self.blockU2 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(64,32,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(32))
		module_list.append(torch.nn.Conv1d(32,32,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(32))
		module_list.append(torch.nn.ConvTranspose1d(32,31,20,stride=2,padding=9))
		module_list.append(torch.nn.ReLU())
		self.blockU1 = torch.nn.ModuleList(module_list)
		
		module_list = []
		module_list.append(torch.nn.Conv1d(32,32,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.BatchNorm1d(32))
		module_list.append(torch.nn.Conv1d(32,32,21,padding=10))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(32,1,1,padding=0))
		self.blockU0 = torch.nn.ModuleList(module_list)
	
	def forward(self, data):
		U0 = data
		for module in self.block1: data = module(data)
		U1 = data
		for module in self.block2: data = module(data)
		U2 = data
		for module in self.block3: data = module(data)
		U3 = data
		for module in self.block4: data = module(data)
		U4 = data
		for module in self.block5: data = module(data)
		U5 = data
		for module in self.block6: data = module(data)
		data = torch.cat((data,U5),dim=1)
		for module in self.blockU5: data = module(data)
		data = torch.cat((data,U4),dim=1)
		for module in self.blockU4: data = module(data)
		data = torch.cat((data,U3),dim=1)
		for module in self.blockU3: data = module(data)
		data = torch.cat((data,U2),dim=1)
		for module in self.blockU2: data = module(data)
		data = torch.cat((data,U1),dim=1)
		for module in self.blockU1: data = module(data)
		data = torch.cat((data,U0),dim=1)
		for module in self.blockU0: data = module(data)
		
		return data

class Autoencoder(torch.nn.Module):
	def __init__(self):
		super().__init__()
			
		module_list = []
		
		module_list.append(torch.nn.Conv1d(1,32,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(32,64,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(64,128,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(128,256,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(256,256,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(256,256,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(256,256,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(256,256,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(256,128,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(128,64,11,padding=5))
		module_list.append(torch.nn.ReLU())
		module_list.append(torch.nn.Conv1d(64,1,11,padding=5))
		
		
		self.network = torch.nn.ModuleList(module_list)
		
		module_list2 = []
		
		module_list2.append(torch.nn.Conv1d(1,32,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.MaxPool1d(2,2))
		module_list2.append(torch.nn.Conv1d(32,64,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.MaxPool1d(2,2))
		module_list2.append(torch.nn.Conv1d(64,128,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.MaxPool1d(2,2))
		module_list2.append(torch.nn.Conv1d(128,256,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.MaxPool1d(2,2))
		module_list2.append(torch.nn.Conv1d(256,512,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.MaxPool1d(2,2))
		module_list2.append(torch.nn.Conv1d(512,1024,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.Upsample(scale_factor=2,mode='nearest'))
		module_list2.append(torch.nn.Conv1d(1024,512,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.Upsample(scale_factor=2,mode='nearest'))
		module_list2.append(torch.nn.Conv1d(512,256,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.Upsample(scale_factor=2,mode='nearest'))
		module_list2.append(torch.nn.Conv1d(256,128,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.Upsample(scale_factor=2,mode='nearest'))
		module_list2.append(torch.nn.Conv1d(128,64,21,padding=10))
		module_list2.append(torch.nn.ReLU())
		module_list2.append(torch.nn.Upsample(scale_factor=2,mode='nearest'))
		module_list2.append(torch.nn.Conv1d(64,1,21,padding=10))
		
		self.pool_network = torch.nn.ModuleList(module_list2)
		
	def forward(self, data):
		x = data.clone()
		for module in self.network:
			x = module(x)
			
		for module in self.pool_network:
			data = module(data)
		return data+x
