#encoding:utf-8

from module.importer import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *

#乱数のシードを設定　これにより再現性を確保できる
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#データセットの、各データへのパスのフォーマット　make_datapath_listへの引数
dataroot = '../../../../dataset_too_large/celeba/images/*.jpg'
#バッチサイズ
batch_size = 128
#入力する乱数の大きさ
z_dim = 20
#エポック数
num_epochs = 5
#optimizerに使う学習率
lr = 0.0002

#訓練データの読み込み、データセット作成
train_img_list = make_datapath_list('../../../../dataset_too_large/celeba/images/*.jpg')
data_transform = transforms.Compose([
				transforms.CenterCrop(160),
				transforms.Resize((64,64)),
				transforms.ToTensor(),
				transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
			])
train_dataset = GAN_Img_Dataset(file_list=train_img_list,transform=data_transform)
dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

#GPUが使用可能かどうか確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

# #ネットワークを初期化するための関数
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		#平均0.0,標準偏差0.02となるように初期化
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

#Generatorのインスタンスを生成
netG = Generator(z_dim=z_dim)
#ネットワークをデバイスに移動
netG = netG.to(device)
#ネットワークの初期化
netG.apply(weights_init)

#Discriminatorのインスタンスを生成
netD = Discriminator(z_dim=z_dim)
#ネットワークをデバイスに移動
netD = netD.to(device)
#ネットワークの初期化
netD.apply(weights_init)

#BCELoss関数の初期化
criterion = nn.BCELoss()
#Generatorの学習過程を見るためのノイズを生成
fixed_noise = torch.randn(64,z_dim,1,1,device=device)
#本物画像、生成画像にそれぞれラベルを設定　損失関数の計算に用いる
real_label = 1.
fake_label = 0.
#Adam optimizersをGeneratorとDiscriminatorに適用
beta1 = 0.5
beta2 = 0.999
optimizerD = optim.Adam(netD.parameters(),lr=lr,betas=(beta1,beta2))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(beta1,beta2))

#学習開始
#学習過程を追うための変数
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training")

#学習開始時刻を保存
t_epoch_start = time.time()
#エポックごとのループ
for epoch in range(num_epochs):
	#データセットからbatch_size枚ずつ取り出し学習
	for i, data in enumerate(dataloader, 0):
		#実際に取得できたバッチサイズを取得
		minibatch_size = data.shape[0]
		#ミニバッチサイズ1だと正規化でエラーになるので避ける
		if(minibatch_size==1): continue
		#data.size() = torch.Size([batch_size,3,64,64])
		data = data.to(device)
		#-------------------------
 		#discriminatorの学習
		#-------------------------
		#損失関数log(D(x)) + log(1 - D(G(z)))を最大化するよう学習する
 		#-------------------------
		######本物画像で学習
		#前のイテレーションでたまった傾きをリセット
		netD.zero_grad()
		#ラベルのデータを作成
		label = torch.full((minibatch_size,),real_label,dtype=torch.float,device=device)
		#minibatch_size枚の本物画像をDiscriminatorに入れ、minibatch_size個分判断結果を得る
		output = netD(data).view(-1)
		#Discriminatorの損失を計算
		errD_real = criterion(output, label)
		#それによる損失関数の傾きを計算
		errD_real.backward()
		#minibatch_size個分の本物画像の判定結果の平均をD_xとする
		D_x = output.mean().item()

		######生成画像で学習
		#ノイズを生成
		noise = torch.randn(minibatch_size,z_dim,1,1,device=device)
		#Generatorからminibatch_size枚画像を出力
		fake_images = netG(noise)
		label.fill_(fake_label)
		#minibatch_size枚の生成画像をDiscriminatorに入れ、minibatch_size個分判断結果を得る
		output = netD(fake_images.detach()).view(-1)
		#Discriminatorの損失を計算
		errD_fake = criterion(output,label)
		#それによる損失関数の傾きを計算
		errD_fake.backward()
		#minibatch_size個分の生成画像の判定結果の平均をD_G_z1とする
		D_G_z1 = output.mean().item()
		#式log(D(x)) + log(1 - D(G(z)))
		errD = errD_real + errD_fake
		#Discriminatorのパラメーターを更新
		optimizerD.step()

		#-------------------------
 		#Generatorの学習
		#-------------------------
		#損失関数log(D(G(z)))を最大化するよう学習する
 		#-------------------------
		netG.zero_grad()
		label.fill_(real_label)  # fake labels are real for generator cost
		#生成画像をDiscriminatorにminibatch_size枚入れ、結果(minibatch_size個分)をoutputとする
		output = netD(fake_images).view(-1)
		#Generatorの損失を計算
		errG = criterion(output, label)
		#それによる損失関数の傾きを計算
		errG.backward()
		#minibatch_size個分の判定結果の平均をD_G_z2とする
		D_G_z2 = output.mean().item()
		#Generatorのパラメーターを更新
		optimizerG.step()

		#学習状況を出力
		if i % 50 == 0:
			print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					% (epoch, num_epochs, i, len(dataloader),
						errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

		#後でグラフに出力する用にlossを記録
		G_losses.append(errG.item())
		D_losses.append(errD.item())

		#Generatorの学習状況を記録
		if (iters % 250 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
			with torch.no_grad():
				fake = netG(fixed_noise).detach().cpu()
			img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

		iters += 1
		#テスト用break
		#break

#学習にかかった時間を出力
#学習終了時の時間を記録
t_epoch_finish = time.time()
total_time = t_epoch_finish - t_epoch_start
with open('./output_img/time.txt', mode='w') as f:
	f.write("total_time: {:.4f} sec.\n".format(total_time))
	f.write("dataset size: {}\n".format(len(train_img_list)))
	f.write("num_epochs: {}\n".format(num_epochs))
	f.write("batch_size: {}\n".format(batch_size))

#本物画像と生成画像の出力
#figオブジェクトから目盛り線などを消す
#batch_size枚の本物画像を取得
real_images = next(iter(dataloader))
#本物画像を出力
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
#表示するには、
#例えば画像が[channel,height,width]となっている時
#transpose(1,2,0)とすることで
#[height,width,channel]に変換する必要がある
plt.imshow(np.transpose(vutils.make_grid(real_images.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
#生成画像を出力　最後のepochでのGeneratorの出力を用いる
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
plt.savefig('./output_img/img.png',dpi=300)

#lossのグラフを出力
plt.clf()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./output_img/loss.png')

#Generatorの学習過程を出力
plt.clf()
fig = plt.figure(figsize=(8,8))
plt.axis("off")
plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
ims = [[plt.imshow(np.transpose(i,(1,2,0)),animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig,ims,interval=200,repeat_delay=1000,blit=True,repeat=False)
ani.save('./output_img/anim.gif', writer="imagemagick")
