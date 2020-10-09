#encoding:utf-8

from .importer import *
from .base_module import *

class Generator(nn.Module):
	def __init__(self,model_size=64,z_dim=100):
		super().__init__()
		self.model_size = model_size #論文内ではdとされている値

		self.full_connection_1 = nn.Linear(z_dim,256*model_size)

		self.layer_1 = nn.Sequential(
				Transpose1dLayer(16*model_size,8*model_size,kernel_size=25,stride=1,upsample=4),
				nn.ReLU(inplace=True))
		self.layer_2 = nn.Sequential(
				Transpose1dLayer( 8*model_size,4*model_size,kernel_size=25,stride=1,upsample=4),
				nn.ReLU(inplace=True))
		self.layer_3 = nn.Sequential(
				Transpose1dLayer( 4*model_size,2*model_size,kernel_size=25,stride=1,upsample=4),
				nn.ReLU(inplace=True))
		self.layer_4 = nn.Sequential(
				Transpose1dLayer( 2*model_size,  model_size,kernel_size=25,stride=1,upsample=4),
				nn.ReLU(inplace=True))
		self.layer_5 = nn.Sequential(
				Transpose1dLayer(   model_size,           1,kernel_size=25,stride=1,upsample=4),
				nn.Tanh())

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
				nn.init.kaiming_normal(m.weight.data)

	def forward(self, x):
		x = self.full_connection_1(x).view(-1,16*self.model_size,16)
		x = F.relu(x)
		x = self.layer_1(x)
		x = self.layer_2(x)
		x = self.layer_3(x)
		x = self.layer_4(x)
		output = self.layer_5(x)
		return output








