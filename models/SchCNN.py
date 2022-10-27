
import os
import sys
sys.path.append(sys.path[0][:-6])
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import config

class SchCNN(nn.Module):
	def __init__(self):
		super().__init__()
		if(config.AUDIO_PROCESSSING_METHOD == '2048_512_128'):
			self.outNum_Conv = 64 * 12 * 1
		elif(config.AUDIO_PROCESSSING_METHOD == '1024_315_80'):
			self.outNum_Conv = 64 * 7 * 11
			# self.outNum_Conv = 32 * 5 * 9
		# (N, 1, 80, 115) / (N, 1, 128, 25)
		self.conv0 = nn.Conv2d(1, 64, kernel_size = (3, 3), stride = (1, 1))
		# (N, 64, 78, 113) / (N, 64, 126, 23)
		self.conv0_follow = nn.Sequential(
			nn.LeakyReLU(0.01),
			nn.Conv2d(64, 32, kernel_size = (3, 3), stride = (1, 1)),
			# (N, 32, 76, 111) / (N, 32, 124, 21)
			nn.LeakyReLU(0.01),
			nn.MaxPool2d((3, 3), (3, 3)),
			# (N, 32, 25, 37) / (N, 32, 41, 7)
			)
		self.conv1 = nn.Sequential(
			# (N, 32, 25, 37) / (N, 32, 41, 7)
			nn.Conv2d(32, 128, kernel_size = (3, 3), stride = (1, 1)),
			# (N, 128, 23, 35) / (N, 128, 39, 5)
			nn.LeakyReLU(0.01),
			nn.Conv2d(128, 64, kernel_size = (3, 3), stride = (1, 1)),
			# (N, 64, 21, 33) / (N, 64, 37, 3)
			nn.LeakyReLU(0.01),
			nn.MaxPool2d((3, 3), (3, 3)),
			# (N, 64, 7, 11) / (N, 64, 12, 1)
			)
		# self.conv2 = nn.Sequential(
		# 	nn.Conv2d(64, 32, kernel_size = (3, 3), stride = (1, 1)),
		# 	# (N, 32, 5, 9) / (N, 64, 37, 3)
		# 	nn.LeakyReLU(0.01),
		# 	)
		self.dense0 = nn.Sequential(
			nn.Dropout(p = config.DROPOUTRATE),
			# Default: 0.5
			nn.Linear(self.outNum_Conv, 256),
			# print(1, '\n'),
			nn.LeakyReLU(0.01),
			nn.Dropout(p = config.DROPOUTRATE),
			nn.Linear(256, 64),
			nn.LeakyReLU(0.01),
			nn.Dropout(p = config.DROPOUTRATE),
			nn.Linear(64, 1),
			# (N, 1)2
			)
		self.sigmoid = nn.Sigmoid()
		# print('#-- Initializing Schluter CNN			--#')
		self.apply(self._init_weights)

	def forward(self, x, T = 1):
		if(config.ZERO_MEAN == True):
			# make zero-mean filter for the use of song-level invariant.
			self.conv0.weight = nn.Parameter(self.conv0.weight - nn.Parameter(torch.mean(self.conv0.weight, dim = (-2, -1), keepdim = True)))
		x = self.conv0(x)
		x = self.conv0_follow(x)
		x = self.conv1(x)
		# x = self.conv2(x)
		# print(x.shape)
		x = torch.flatten(x, start_dim=1, end_dim=-1) 
		x = self.dense0(x)
		x = torch.div(x, T)
		x = self.sigmoid(x)
		x = x.view(-1)
		# x = torch.squeeze(x)
		return x

	def _init_weights(self, m) -> None:
		# As is, it does absolutely nothing. It is a type annotation for 
		# the main function that simply states that this function returns None.
		if(isinstance(m, nn.Conv2d)):
			nn.init.kaiming_uniform_(m.weight)
		elif(isinstance(m, nn.Linear)):
			nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
	net = SchCNN()
	print(net)
	net = net.cuda()
	summary(net, (1, 80, 115))
	
