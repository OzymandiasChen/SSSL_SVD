
import os
import sys
sys.path.append(sys.path[0][:-6])
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import config


class ResidualBlock(nn.Module):
	def __init__(self, inchannel, outchannel):
		super(ResidualBlock, self).__init__()
		if(inchannel == outchannel):
			kernel_size, stride, padding = (3, 3), (1, 1), (1, 1)
		else:	
			# if channle doubled, feature map size halved.
			kernel_size, stride, padding = (5, 3), (2, 2), (2, 1)
		self.left = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, kernel_size = kernel_size, stride = stride, padding = padding, bias = True),
			nn.BatchNorm2d(outchannel),
			nn.ReLU(inplace=True),
			nn.Conv2d(outchannel, outchannel, kernel_size = (3, 1), stride = (1, 1) , padding = (1, 0), bias=True),
			nn.BatchNorm2d(outchannel)
		)
		self.shortcut = nn.Sequential()
		if inchannel != outchannel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inchannel, outchannel, kernel_size=1, stride = stride, padding = 0, bias=False),
				nn.BatchNorm2d(outchannel)
			)

	def forward(self, x):
		out = self.left(x)
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Res_ZeroMean_StdPool(nn.Module):
	def __init__(self, num_classes = 2):
		super().__init__()
		# (N, 1, 128, 50)
		self.conv1 = nn.Conv2d(1, 32, kernel_size = (5, 3), stride = (2, 1), padding = (2, 1), bias = True) # weight.shape = (32, 1, 5， 3)
		self.conv1_f = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(), )
		# (N, 32, 64, 50)
		self.conv2_1 = self.make_resLayer(32, 32)
		self.conv2_2 = self.make_resLayer(32, 32)
		# (N, 32, 64, 50) -> (N, 32, 64, 50)
		# self.conv3_1 = self.make_resLayer(32, 64)
		# self.conv3_2 = self.make_resLayer(64, 64)
		# (N, 64, 32, 25) -> (N, 64, 32, 25)
		# self.conv4_1 = self.make_resLayer(64, 128)
		# self.conv4_2 = self.make_resLayer(128, 128)
		# # (N, 128, 16, 13) -> (N, 128, 16, 13)
		# self.conv5_1 = self.make_resLayer(128, 256)
		# self.conv5_2 = self.make_resLayer(256, 256)
		# (N, 256, 8, 7) -> (N, 256, 8, 7)
		self.gap = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(32, 1)
		self.sigmoid = nn.Sigmoid()
		self.apply(self._init_weights)

	def forward(self, x, T = 1):
		# make zero-mean filter for the use of song-level invariant.
		self.conv1.weight = nn.Parameter(self.conv1.weight - nn.Parameter(torch.mean(self.conv1.weight, dim = (-2, -1), keepdim = True)))
		x = self.conv1(x)
		x = self.conv1_f(x)
		x = self.conv2_1(x)
		x = self.conv2_2(x)
		# x = torch.flatten(x, start_dim=1, end_dim=-1) 
		# print(x.shape)
		# x = self.dense0(x)
		# x = torch.div(x, T)
		# x = self.sigmoid(x)
		# x = x.view(-1)
		# x = self.conv3_1(x)
		# x = self.conv3_2(x)
		# x = self.conv4_1(x)
		# x = self.conv4_2(x)
		# x = self.conv5_1(x)
		# x = self.conv5_2(x)
		x = self.gap(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		x = torch.div(x, T)
		x = self.sigmoid(x)
		x = x.view(-1)
		return x

	def make_resLayer(self, inchannel, outchannel):
		layers = []
		layers.append(ResidualBlock(inchannel, outchannel))
		return nn.Sequential(*layers)

	def _init_weights(self, m) -> None:
		# As is, it does absolutely nothing. It is a type annotation for 
		# the main function that simply states that this function returns None.
		if(isinstance(m, nn.Conv2d)):
			nn.init.kaiming_uniform_(m.weight)
		elif(isinstance(m, nn.Linear)):
			nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
	# m.weight = nn.Parameter(m.weight -torch.nn.Parameter(torch.mean(m.weight, dim=(-2, -1), keepdim = True)))
	# m.weight
	# m.bias
	net = Res_ZeroMean_StdPool()
	print(net)
	net = net.cuda()
	summary(net, (1, 80, 115))
	
