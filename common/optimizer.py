import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    # 書籍1ではparamsやgradsは辞書だったのでkeyについてループしてvalueを更新していたが、今回はリストなのでindexで指定して要素を更新する。
    # params[key]とparams[i]は同じものを指している。
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
