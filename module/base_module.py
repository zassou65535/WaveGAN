#encoding:utf-8

from .importer import *

class Transpose1dLayer(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size,stride,padding=11,upsample=None,output_padding=1):
		super().__init__()
		self.upsample = upsample

		self.conv_layer_1 = nn.Sequential(
				nn.Upsample(scale_factor=upsample),
				nn.ConstantPad1d(kernel_size//2,value=0),
				nn.Conv1d(in_channels,out_channels,kernel_size,stride)
				)

		self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

	def forward(self, x):
		if self.upsample:
			return self.conv_layer_1(x)
		else:
			return self.Conv1dTrans(x)
