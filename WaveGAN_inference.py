#encoding:utf-8

from module.importer import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *

#出力する音声ファイルの数
sample_size = 16
#入力する乱数の次元の大きさ
z_dim = 20
#扱う音声のサンプリングレート
sampling_rate = 16000

#学習済みモデルの読み込み
netG = Generator(z_dim=z_dim)
trained_model_path = "./output/generator_trained_model_cpu.pth"
netG.load_state_dict(torch.load(trained_model_path))
#推論モードに切り替え
netG.eval()
#ノイズ生成
noise = torch.Tensor(sample_size,z_dim).uniform_(-1,1)
#generatorへ入力、出力画像を得る
generated_sound = netG(noise)
#出力用ディレクトリがなければ作成
output_dir = "./output/inference"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
#音声ファイルの出力
save_sounds("./output/inference/",generated_sound,sampling_rate)
