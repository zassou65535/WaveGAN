#encoding:utf-8

from .importer import *
from .base_module import *

class Discriminator(nn.Module):
	def __init__(self,model_size=64,shift_factor=2):
		super().__init__()
		self.model_size = model_size #論文内ではdとされている値
		self.shift_factor = shift_factor  #n　どれだけ揺さぶりをかけるか

		self.layer_1 = nn.Sequential(
				nn.Conv1d(           1,   model_size,kernel_size=25,stride=4,padding=11),
				nn.LeakyReLU(0.2,inplace=True),
				PhaseShuffle(shift_factor)
				)
		self.layer_2 = nn.Sequential(
				nn.Conv1d(  model_size, 2*model_size,kernel_size=25,stride=4,padding=11),
				nn.LeakyReLU(0.2,inplace=True),
				PhaseShuffle(shift_factor)
				)
		self.layer_3 = nn.Sequential(
				nn.Conv1d(2*model_size, 4*model_size,kernel_size=25,stride=4,padding=11),
				nn.LeakyReLU(0.2,inplace=True),
				PhaseShuffle(shift_factor)
				)
		self.layer_4 = nn.Sequential(
				nn.Conv1d(4*model_size, 8*model_size,kernel_size=25,stride=4,padding=11),
				nn.LeakyReLU(0.2,inplace=True),
				PhaseShuffle(shift_factor)
				)
		self.layer_5 = nn.Sequential(
				nn.Conv1d(8*model_size,16*model_size,kernel_size=25,stride=4,padding=11),
				nn.LeakyReLU(0.2,inplace=True),
				PhaseShuffle(shift_factor)
				)

		self.full_connection_1 = nn.Linear(256*model_size,1)

		for m in self.modules():
			if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
				nn.init.kaiming_normal(m.weight.data)

	def forward(self, x):
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.layer_3(x)
		x = self.layer_4(x)
		x = self.layer_5(x)
		x = x.view(-1,256*self.model_size)
		output = self.full_connection_1(x)
		return output

