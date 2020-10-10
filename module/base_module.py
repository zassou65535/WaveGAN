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



#discriminatorの損失関数の、勾配制約項の計算に必要な関数「gradient_penalty」
#WGAN-GPにおいて、discriminatorの損失関数は　E[本物の音声の判定結果]-E[偽音声の判定結果]+勾配制約項　と表され、
#generatorでは、E[偽音声の判定結果]と表される
def gradient_penalty(netD,real,fake,batch_size,gamma=1):
	device = real.device
	#requires_gradが有効なTensorに対してはbackwardメソッドが呼べて、自動的に微分を計算できる
	alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(device)
	#本物画像と偽画像を任意の割合で混ぜ合わせる
	x = alpha*real + (1-alpha)*fake
	#それをdiscriminatorに入れ、結果をd_とする
	d_ = netD.forward(x)
	#出力d_と入力xから傾きを求める
	#傾きから計算されるL2ノルムが1になると良い結果を生むことが知られている
	#よってこれが1に近づくような学習ができるようにgradient_penaltyを計算
	g = torch.autograd.grad(outputs=d_, inputs=x,
							grad_outputs=torch.ones(d_.shape).to(device),
							create_graph=True, retain_graph=True,only_inputs=True)[0]
	g = g.reshape(batch_size, -1)
	return ((g.norm(2,dim=1)/gamma-1.0)**2).mean()


