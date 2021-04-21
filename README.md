# WaveGAN
## 概要
機械学習で音声を生成する手法「WaveGAN」の実装です。  
参考資料:https://arxiv.org/abs/1802.04208  
解説記事:https://qiita.com/zassou65535/items/5a9d5ef44dedea94be8a  

## 想定環境
python 3.7.1  
`pip install -r requirements.txt`で環境を揃えることができます。 

## プログラム
* `WaveGAN_train.py`は学習を実行し、過程と結果を出力するプログラムです。 
	* 学習においては各イテレーションごとにデータセットから音声(`.wav`形式)を選び出し、約4秒分ランダムな箇所から切り取って学習します。 
* `WaveGAN_inference.py`は`WaveGAN_train.py`が出力した学習結果(重み)をGeneratorに読み込み推論を実行、音声データを出力するプログラムです。 
	* 出力されるwavファイルは約4秒の長さの音声です。 

## 使い方
1. `WaveGAN_train.py`のあるディレクトリに`./dataset`ディレクトリを作成します
1. `./dataset`ディレクトリに、学習に使いたい音声ファイルを`./dataset/**/*.wav`という形式で好きな数入れます
1. `WaveGAN_train.py`の置いてあるディレクトリで`python WaveGAN_train.py`を実行して学習を開始します
	* 学習の過程が`./output/train/`以下に出力されます
	* 学習結果が`./output/generator_trained_model_cpu.pth`として出力されます
1. `WaveGAN_inference.py`の置いてあるディレクトリで`python WaveGAN_inference.py`を実行して推論します
	* 推論結果が`./output/inference/`以下に出力されます
	* 注意点として、`./output/generator_trained_model_cpu.pth`(学習済みモデル)がなければエラーとなります

学習には環境によっては12時間以上要する場合があります。   
