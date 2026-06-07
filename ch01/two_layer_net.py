import sys
import os
import numpy as np

sys.path.append(os.pardir)
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # 重みの初期化
        # 書籍1では辞書のvalueとして保持していたがこちらでは直で保持する
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # レイヤの生成。レイヤのインスタンスのリストを保持する。
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]

        # 最終層は他のレイヤと扱いが異なる（損失計算に絡む）ので別のインスタンス変数に保持する。
        # 具体的には、正解ラベルtが必要なことや、予測時には不要（生のスコアを見ればいいため）であることが違いとしてあげられる。
        self.loss_layer = SoftmaxWithLoss()

        # 各レイヤーがparamsとgradsを持つので、下記のようにシンプルな処理でnet側のparamsも書ける
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):

        # 書籍1では辞書で保持していたが、書籍2ではlayersはlistであることに注意
        # また、softmaxwithlossを切り離しているので下記のように単純な形で書ける。
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def forward(self, x, t):
        # 書籍1におけるlossメソッドとほぼ同じ
        # softmaxwithlossの手前までを計算
        score = self.predict(x)
        # softmaxwithlossまで計算
        loss = self.loss_layer.forward(score, t)
        return loss

    # BPを使用したgradient計算用のメソッド。各レイヤーのbackwardを呼ぶことで勝手にレイヤー側でgradsが更新され、さらにそれとメモリ部分を共有しているモデル側のgradsも更新される。
    def backward(self, dout=1):
        """
        各レイヤーを部品としてうまく分離して実装しているため、単純なfor文でforwardとbackwardを行うことができる。
        """

        # 書籍1ではgradientというメソッドになっており、その中に順伝播も逆伝播も記述していた。

        dout = self.loss_layer.backward(dout)

        # 入出力のshapeは対応するようになっているので下記のような単純なfor文でok
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # 書籍1ではtwo layer net側にパラのgradsを保持していたのでそれを上書きしていたが、今回はレイヤー側と共有しており、レイヤー側で更新されているので不要。
        return dout
