import numpy as np


def softmax(x):
    """
    通常softmax関数は各行ごとに正規化したいので、このような実装にする
    """
    # keepdims=Trueにすることで例えば通常二次元配列に対してaxis指定してmaxを求めると一次元ベクトルになってしまうが、それを元の二次元配列のままにしてくれる。
    c = np.max(
        x, axis=-1, keepdims=True
    )  # axis=-1は最後の軸。二次元配列なら1列に潰す。
    exp_x = np.exp(x - c)  # オーバーフロー対策
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    y = exp_x / sum_exp_x
    return y

def softmax_sample(x):
    


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)  # argmaxを使いたいのでここではtも必ず二次元配列にする

    # one-hotの場合
    if t.size == y.size:
        t = t.argmax(axis=1)  # ラベルを格納した一次元配列が返る
    else:
        t = t.reshape(-1)  # one-hotでない場合も必ず一次元配列にする

    delta = 1e-7

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
