#encoding:utf-8

from .importer import *
from .base_module import *

class Generator(nn.Module):
	def __init__(self,model_size=32,z_dim=20):
		super().__init__()
		self.model_size = model_size #論文内ではdとされている値

		self.full_connection_1 = nn.Linear(z_dim,512*model_size)

		self.layer_1 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=32*model_size,\
									out_channels=16*model_size,\
									kernel_size=25,\
									stride=4,\
									padding=11,\
									output_padding=1),
				nn.ReLU(inplace=True))
		self.layer_2 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=16*model_size,\
									out_channels=8*model_size,\
									kernel_size=25,\
									stride=4,\
									padding=11,\
									output_padding=1),
				nn.ReLU(inplace=True))
		self.layer_3 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=  8*model_size,\
									out_channels=4*model_size,\
									kernel_size=25,\
									stride=4,\
									padding=11,\
									output_padding=1),
				nn.ReLU(inplace=True))
		self.layer_4 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=  4*model_size,\
									out_channels=2*model_size,\
									kernel_size=25,\
									stride=4,\
									padding=11,\
									output_padding=1),
				nn.ReLU(inplace=True))
		self.layer_5 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=  2*model_size,\
									out_channels=  model_size,\
									kernel_size=25,\
									stride=4,\
									padding=11,\
									output_padding=1),
				nn.ReLU(inplace=True))
		self.layer_6 = nn.Sequential(
				nn.ConvTranspose1d(in_channels=model_size,\
									out_channels=1,\
									kernel_size=25,\
									stride=4,\
									padding=11,\
									output_padding=1),
				nn.Tanh())

	def forward(self, x):
		x = self.full_connection_1(x).view(-1,32*self.model_size,16)
		x = F.relu(x)
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.layer_3(x)
		x = self.layer_4(x)
		x = self.layer_5(x)
		output = self.layer_6(x)
		return output








