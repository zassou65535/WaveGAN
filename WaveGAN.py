#encoding:utf-8

from module.importer import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *

#データセットの、各データへのパスのフォーマット　make_datapath_listへの引数
dataset_path = './dataset/**/*.wav'
#バッチサイズ
batch_size = 64
#入力する乱数の大きさ
z_dim = 100
#エポック数
num_epochs = 1
#optimizerに使う学習率
lr = 0.0002
#入力、出力する音声のサンプリングレート
sampling_rate = 16000
#Generatorの学習一回につき、Discriminatorを何回学習させるか
D_updates_per_G_update = 5

#GPUが使用可能かどうか確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

#訓練データの読み込み、データセット作成
train_sound_list = make_datapath_list(dataset_path)
train_dataset = GAN_Sound_Dataset(file_list=train_sound_list,device=device,batch_size=batch_size)
dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

# #ネットワークを初期化するための関数
def weights_init(m):
	if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m,nn.Linear):
		nn.init.kaiming_normal_(m.weight.data)

#Generatorのインスタンスを生成
netG = Generator(z_dim=z_dim)
#ネットワークをデバイスに移動
netG = netG.to(device)
#ネットワークの初期化
netG.apply(weights_init)

#Discriminatorのインスタンスを生成
netD = Discriminator()
#ネットワークをデバイスに移動
netD = netD.to(device)
#ネットワークの初期化
netD.apply(weights_init)

#最適化手法を設定　Adamにする
beta1 = 0.5
beta2 = 0.9
optimizerD = optim.Adam(netD.parameters(),lr=lr,betas=(beta1,beta2))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(beta1,beta2))

#学習開始
#学習過程を追うための変数
G_losses = []
D_losses = []
iters = 0

print("Starting Training")

#学習開始時刻を保存
t_epoch_start = time.time()
#エポックごとのループ
for epoch in range(num_epochs):
	#データセットからbatch_size個ずつ取り出し学習
	for i,real_sound in enumerate(dataloader, 0):
		#実際に取り出せた音声データの数
		minibatch_size = real_sound.shape[0]
		#取り出したミニバッチ数が1の場合勾配を求める過程でエラーとなるので処理を飛ばす
		if(minibatch_size==1): continue
		#GPUが使えるならGPUへ転送
		real_sound = real_sound.to(device)

		#-------------------------
 		#Generatorの学習
		#-------------------------
		#損失関数　-E[偽音声の判定結果]　を最大化するよう学習する
 		#-------------------------
		#ノイズを生成
		z = torch.randn(batch_size,z_dim).to(device)
		#ノイズをgeneratorに入力、出力音声をfake_soundとする
		fake_sound = netG.forward(z)
		#出力音声fake_soundをdiscriminatorで推論　つまり偽音声の入力をする
		d_ = netD.forward(fake_sound)

		# WGAN_GPではミニバッチ内の推論結果全てに対し平均を取り、それを誤差伝搬に使う
		errG = -d_.mean()#E[偽音声の判定結果]を計算
		#前のイテレーションで計算した傾きが残ってしまっているのでそれをリセットしておく
		optimizerG.zero_grad()
		#損失の傾きを計算して
		errG.backward()
		#実際に誤差伝搬を行う
		optimizerG.step()

		#-------------------------
 		#discriminatorの学習
		#-------------------------
		#損失関数　E[本物の音声の判定結果]-E[偽音声の判定結果]+勾配制約項　を最大化するよう学習する
		#Generatorの学習1回につき、D_updates_per_G_update回Discriminatorを学習する
 		#-------------------------
		errD_loss_sum = 0#Discriminator学習時の、損失の平均を取る用の変数
		for _ in range(D_updates_per_G_update):
			#ノイズを生成、zとする
			z = torch.randn(minibatch_size,z_dim).to(device)
			#generatorにノイズを入れ偽音声を生成、fake_soundとする
			fake_sound = netG.forward(z)
			#本物の音声を判定、結果をdに格納
			d = netD.forward(real_sound)
			#偽音声を判定、結果をd_に格納
			d_ = netD.forward(fake_sound)

			#ミニバッチごとの、判定結果の平均をそれぞれとる
			loss_real = d.mean()#-E[本物の音声の判定結果]を計算
			loss_fake = d_.mean()#-E[偽音声の判定結果]を計算
			#勾配制約項の計算
			loss_gp = gradient_penalty(netD,real_sound.data,fake_sound.data,minibatch_size)
			beta_gp = 10.0
			#E[本物の音声の判定結果]-E[偽音声の判定結果]+勾配制約項 を計算
			errD = -loss_real + loss_fake + beta_gp*loss_gp
			#前のイテレーションで計算した傾きが残ってしまっているのでそれをリセットしておく
			optimizerD.zero_grad()
			#損失の傾きを計算して
			errD.backward()
			#実際に誤差伝搬を行う
			optimizerD.step()
			#後で平均を取るためにlossを記録
			errD_loss_sum += errD.item()

		#学習状況を出力
		if i % 50 == 0:
			print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
					% (epoch, num_epochs, i, len(dataloader),
						errD_loss_sum/D_updates_per_G_update, errG.item()))

		#後でグラフに出力する用にlossを記録
		G_losses.append(errG.item())
		D_losses.append(errD_loss_sum/D_updates_per_G_update)

		iters += 1
		#テスト用break
		#break

#-------------------------
#実行結果の出力
#-------------------------

#ディレクトリ「./output」がなければ作成
if not os.path.exists("./output"):
	os.makedirs("./output")

#生成された音声の出力
generating_num = 20#音声をいくつ出力したいか
z = torch.randn(generating_num,z_dim).to(device)
generated_sound = netG(z)
for i,sound in enumerate(generated_sound):
	print(type(sound))
	print(sound.shape)
	sound = sound.squeeze(0)
	print(sound.shape)
	sound = sound.to('cpu').detach().numpy().copy()
	librosa.output.write_wav("./output/generated_sound_{}.wav".format(i+1),sound,sampling_rate)

#学習にかかった時間を出力
#学習終了時の時間を記録
t_epoch_finish = time.time()
total_time = t_epoch_finish - t_epoch_start
with open('./output/time.txt', mode='w') as f:
	f.write("total_time: {:.4f} sec.\n".format(total_time))
	f.write("dataset size: {}\n".format(len(train_sound_list)))
	f.write("num_epochs: {}\n".format(num_epochs))
	f.write("batch_size: {}\n".format(batch_size))

#lossのグラフを出力
plt.clf()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./output/loss.png')

