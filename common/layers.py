import numpy as np
from common.functions import softmax, cross_entropy_error

# 各レイヤはメソッドとしてforward(), backward()を持つ。
# 各レイヤはインスタンス変数としてparams, gradsを持つ。
# 書籍1ではnet側でparamsを辞書型のインスタンス変数として持ち、それら（のvalue）を使ってレイヤーのインスタンス変数を初期化していた。（レイヤー側ではWやbなどの単位でインスタンス変数として持っていた）
# 書籍2ではレイヤー側の時点でWやbをまとめてリストとし、paramsというインスタンス変数で保持する。


class Sigmoid:
    def __init__(self):
        # sigmoidに学習可能なパラメータは存在しないので空のリストで初期化。
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        # backwardの計算で必要なので保持しておく
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    """テンソル対応。書籍のは二次元配列前提となっている。"""

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

        self.original_x_shape = None

    def forward(self, x):
        """
        テンソル対応でflattenのような操作を入れる
        """

        W, b = self.params

        # テンソル対応
        # 例えばカラー画像なら4階のテンソルなので(10,3,32,32)のようなshape
        self.original_x_shape = x.shape

        # ここでflatttenのような操作を入れる。各データを一本の横ベクトルにするイメージ
        # x.shape[0]は常にデータ数(バッチ数)→つまりデータ数分の行数があり、各行は一本のflattenされた横ベクトルとなっている行列になる。
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, W) + b

        return out

    def backward(self, dout):
        """

        Affineの内部計算の都合で2次元に変形するが、外から見た入出力の形は壊してはいけないので元に戻す
        """
        W, b = self.params

        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        # 元のデータ形状に戻す。これは単に偏微分の並べ方を変えているだけなので問題ない（見かけを変えているだけ）
        # shapeはタプルになっているのでアンパックして引数として渡しているだけ
        dx = dx.reshape(*self.original_x_shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        # backwardの処理が単純なfor文でいいようにdWやdbは保持するだけで出力はしない。（下流に伝える必要があるものだけにする）
        # 各レイヤーを一つの演算のように扱っているので、基本的に下流に伝える偏微分は一つだけで済む。
        return dx


class SoftmaxWithLoss:
    """
    x,y,tはバッチを想定していることに注意
    """

    def __init__(self):
        # 学習可能なパラはないので空
        self.params, self.grads = [], []
        self.loss = None
        # 下記二つはbackwardの計算時に必要
        self.y = None
        self.t = None

    def forward(self, x, t):

        # サンプルコードではデータの形状を変換しているが、common.functionsの関数側で対応しているので不要なはず？
        self.t = t
        self.y = softmax(x)
        # backwardの計算自体にはlossは不要だが、便利なので保持しておく
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        # 教師データがone-hot（つまり出力と同じ形のベクトル）
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size

        else:
            dx = self.y.copy()
            # バッチの各行について、正解のindexのみ1を引く
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        (W,) = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        (W,) = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # 普通にgrads[0]=dWだとdWが参照しているオブジェクトをgrads[0]も参照するようになる
        self.grads[0][...] = dW
        return dx
