import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
	def forward(self, x):
		N, C, H, W = x.size()
		return x.view(N, -1)

class SiameseNet(nn.Module):
	"""docstring for SiameseNet"""
	def __init__(self):
		super(SiameseNet, self).__init__()
		self.siamese_twin = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 10),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 7),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 4),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4),
			nn.ReLU(),
			Flatten(),
			nn.Linear(in_features = 256 * 6 * 6 , out_features = 4096),
			nn.Sigmoid()
			)
		self.final_fc = nn.Linear(in_features = 4096 , out_features = 1)

	def forward(self, x, y):
		x1 = self.siamese_twin(x)
		y1 = self.siamese_twin(y)
		L1 = torch.abs(x1 - y1)
		output = F.sigmoid(self.final_fc(L1))

		return output


		


