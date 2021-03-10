#encoding:utf-8

from .importer import *

class PhaseShuffle(nn.Module):
	#PhaseShuffleを行うレイヤーの定義
	def __init__(self,n):
		super().__init__()
		self.n = n#どれだけずらすかの範囲は論文内では[-n,n]と定義されている

	def forward(self, x):
		#nが0であれば、PhaseShuffleをそもそもしないのと同等
		if self.n == 0:
			return x
		#[-n,n]に属する整数をランダムに生成、shiftとする
		shift = torch.Tensor(x.shape[0]).random_(-self.n,self.n+1).type(torch.int)
		#xにPhaseShuffleを適用した結果をx_shuffledに格納、戻り値とする
		x_shuffled = x.clone()
		for i,shift_num in enumerate(shift):
			if(shift_num==0): continue
			dim = len(x_shuffled[i].size()) - 1
			origin_length = x[i].shape[dim]
			if shift_num > 0:
				left = torch.flip(torch.narrow(x[i],dim,1,shift_num),[dim])
				right = torch.narrow(x[i],dim,0,origin_length-shift_num)
			else:
				shift_num = -shift_num
				left = torch.narrow(x[i],dim,shift_num,origin_length-shift_num)
				right = torch.flip(torch.narrow(x[i],dim,origin_length-shift_num-1,shift_num),[dim])
			x_shuffled[i] = torch.cat([left,right],dim)

		return x_shuffled

#discriminatorの損失関数の、勾配制約項の計算に必要な関数「gradient_penalty」を求める関数
#WGAN-GPにおいて、discriminatorの損失関数は　E[本物の音声の判定結果]-E[偽音声の判定結果]+勾配制約項　と表され、
#generatorでは、E[偽音声の判定結果]と表される
def gradient_penalty(netD,real,fake,batch_size,gamma=1):
	device = real.device
	#requires_gradが有効なTensorに対してはbackwardメソッドが呼べて、自動的に微分を計算できる
	alpha = torch.rand(batch_size,1,1,requires_grad=True).to(device)
	#本物と偽物をランダムな割合で混ぜ合わせる
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


