import sys
import os
import numpy as np
import collections

sys.path.append(os.pardir)
from common.layers import Embedding
from ch04.negative_samling_layer import NegativeSamplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        # レイヤの生成
        self.in_layers = []
        # W_in側のembedding layerを両側ウィンドウサイズ分生成
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        # 下記については、NegativeSamplingLossクラスの中でW_outを共有する正例+負例サンプリング分のレイヤが生成されている
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 全ての重みと勾配をリストにまとめる
        # W_inのレイヤインスタンスが2ウィンドウサイズ分+NegativeSamplingLossレイヤ一つ分のリスト
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 分散表現を明示的にインスタンス変数として保持する
        self.word_vecs = W_in

    def forward(self, contexts, target):
        # contextsは各行が2ウィンドウサイズ分の単語が並んだベクトルになっている。
        h = 0
        # ウィンドウ内の各単語に対応するembeddingレイヤについて周辺単語のベクトルを出して平均を取るために一旦加算する
        for i, layer in enumerate(self.in_layers):
            # 各行のi番目の単語を抜き出した列ベクトルに対してforwardを適用しているので
            # hは各行が各バッチにおける埋め込みの和になっている
            h += layer.forward(contexts[:, i])
        # 2ウィンドウサイズで割ることで、各行が平均ベクトル（文脈ベクトル）になる
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        # この時点では各レイヤで勾配を持っており、trainerのremove dupで文脈を跨いだ加算が行われるイメージ
        for layer in self.in_layers:
            layer.backward(dout)

        return None
