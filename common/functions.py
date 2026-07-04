import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    """
    通常softmax関数は各行ごとに正規化したいので、このような実装にする
    """
    # keepdims=Trueにする理由：例えば通常二次元配列に対してaxis指定してmaxを求めると一次元ベクトルになってしまうが、それを元の二次元配列のままにしてくれる。
    # そうすることで、各行に対して最大値が算出され、各行に対してその行の最大値を引くような挙動になる。
    c = np.max(
        x, axis=-1, keepdims=True
    )  # axis=-1は最後の軸。二次元配列なら1列に潰す。
    exp_x = np.exp(x - c)  # オーバーフロー対策
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    y = exp_x / sum_exp_x
    return y


def softmax_sample(x):
    """書籍に載っているのは下記のように明示的にndimが1と2の場合を処理する書き方

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)  # オーバーフロー対策
        x = np.exp(x) / np.sum(np.exp(x))

    return x


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
