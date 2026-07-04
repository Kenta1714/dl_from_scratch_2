import numpy as np
from common.layers import Embedding
from common.functions import softmax, sigmoid


class RNN:
    """1ステップ分の処理を行う"""

    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        """1ステップ前の隠れ状態と現在の入力から、次に渡す隠れ状態を作る。

        Args:
            x (_type_): (N,D)の入力
            h_prev (_type_): _description_

        Returns:
            _type_: _description_
        """
        Wx, Wh, b = self.params
        # 線形変換
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        # 活性化関数
        h_next = np.tanh(t)

        # 逆伝播で使用する
        self.cache = (x, h_prev, h_next)

        return h_next

    def backward(self, dh_next):
        """_summary_

        Args:
            dh_next (_type_): すでに出力側と次のセルに渡した隠れ状態側の勾配が合算されて勾配として渡される

        Returns:
            _type_: _description_
        """
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # tanhの微分は1-tanh^2より、それを上流からのdh_nextにかける
        dt = dh_next * (1 - h_next**2)
        # 分岐ノードなので、各バッチの分を足し合わせる
        db = np.sum(dt, axis=0)
        # dtがMatMulまですり抜けるので、単純にMatMulの逆伝播
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        # 下記の二つはセルの外側から来ているので勾配を流す
        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        """_summary_

        Args:
            Wx (_type_): _description_
            Wh (_type_): _description_
            b (_type_): _description_
            stateful (bool, optional): 隠れ状態を保持するか。Falseの場合forwardを呼ぶたびにゼロ行列で初期化. Defaults to False.
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        # 複数のRNNレイヤをリストで管理する用
        self.layers = None

        # hは最後のRNNレイヤの隠れ状態。TimeRNNレイヤ間の順伝播の引き継ぎに使う。書籍例でいえば最初の10ブロックから次の10ブロックへのhの引き継ぎに使う。
        # さらにforループ中では、一つのTimeRNNレイヤの中での、RNNレイヤ同士の引き継ぎにも使う。

        # dhは基本的には一つのTimeRNNレイヤの中でのセル同士（RNNレイヤ同士）の引き継ぎに使う？（ブロックをまたがないため。）ただし最後のdhをインスタンス変数を保持するようになっている？
        self.h, self.dh = None, None
        # 隠れ状態を引き継ぐかどうか。
        self.stateful = stateful

    def forward(self, xs):
        """_summary_

        Args:
            xs (_type_): T*Dの行列（D次元の横ベクトルがタイムステップT分並んだもの）がバッチサイズN個重なったテンソル

        Returns:
            _type_: _description_
        """

        # パラは共有する
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        # 各時刻の隠れ状態を全部保存する配列（基本的にseq2seq的なタスクを想定しており、これが出力系列のイメージ）
        hs = np.empty((N, T, H), dtype="f")

        # statefulでない場合やhがセットされていない場合はゼロ行列で初期化（ただし、そのセルにおいてxWとbが足されるので、次のセルに伝えるhはゼロ行列ではなくなる）
        # また、通常は前のブロックで格納されたself.hが残っているのでそれをそのまま使用するイメージ。
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")

        for t in range(T):
            # 各時刻のRNNはパラメータは共通だがcacheは異なる。よってパラを共有するがレイヤのインスタンスとしては毎回別のものを生成する。
            layer = RNN(*self.params)
            # xs[:, t, :]で時刻tの入力のみを取り出す。shapeは(N,D)

            # ここは理論上はローカル変数hの上書きでforループを回して最後にself.hに格納でも良い
            # また、これにより毎回のhが上書きされてしまうがRNNレイヤ側のcacheに保持されているので逆伝播は普通にできる。
            self.h = layer.forward(xs[:, t, :], self.h)
            # 現在時刻の隠れ状態を出力用配列hsの時刻tの部分に保存する
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype="f")

        # 未来ブロックからの情報（勾配）は打ち切って0とする。（self.dhとかは使わない）
        dh = 0
        # Wx,Wh,bそれぞれの勾配に対応
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            # layersは単純に各tのRNNレイヤインスタンスがT個分格納されたlist
            layer = self.layers[t]
            # 前者は出力側から来る勾配で、後者は隠れ状態として渡した側から（つまり一つ次のセルから）来る勾配。
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            # 3つの重み行列についてループ
            for i, grad in enumerate(layer.grads):
                # 左辺はTimeRNNレイヤのgradで、右辺はRNNレイヤのgrad。つまり各RNNレイヤの勾配をTimeRNN側で累積加算しているイメージ
                grads[i] += grad

        # 下記は単にbackwardメソッドで使う一時的なローカル変数gradsをインスタンス変数self.gradsにコピーしているだけ。
        for i, grad in enumerate(grads):
            # メモリの参照先は変えたくないので[...]を使用する
            self.grads[i][...] = grad

        # 下記の勾配は次のブロックにはTruncatedBPTTの場合伝えないので最悪なくても大丈夫（ただし7章で使うらしい）
        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layer = None
        self.W = W

    def forward(self, xs):

        # xsは単語ベクトルではなくIDが並んだものなので2次元配列。
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype="f")
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]  # Embeddingレイヤのgradsリストは要素が一つだけ

        # 勾配を累積加算したローカル変数gradをself.gradsにコピーする
        self.grads[0][...] = grad
        return None


# TODO 下記はコピペ


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


# TODO コピペ


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = ts != self.ignore_label

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx
