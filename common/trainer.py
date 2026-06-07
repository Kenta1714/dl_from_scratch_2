import sys
import os
import numpy
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from dataset import spiral
from common.optimizer import SGD
from common.utill import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        # ミニバッチを何回繰り返したら1エポックになるか
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()

        for epoch in range(max_epoch):

            # indexをランダムに並び替える（2回目以降はランダムな状態から別のランダムな状態に並び替えられるので問題ない）
            idx = np.random.permutation(
                data_size
            )  # 0からdata_size-1までの数字がランダムに並ぶ
            x = x[idx]
            t = t[idx]

            # 1エポック分までミニバッチを繰り返す。
            for iters in range(max_iters):
                # すでに並べ替えてあるので、先頭からバッチサイズ分ずつ取り出すだけで良い。
                batch_x = x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = t[iters * batch_size : (iters + 1) * batch_size]

                loss = model.forward(batch_x, batch_t)
                # 下記によりgradsが更新される。
                model.backward()
                # 共有された重みを一つに集約（？）
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)

                # modelクラスのインスタンス変数じゃなくて上記で定義した変数を使用している（？）
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) & (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(
                        "| epoch %d | iter %d / %d | time %d[s] | loss %.2f "
                        % (
                            self.current_epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            avg_loss,
                        )
                    )
                    self.loss_list.append(avg_loss)
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    # 以下コピペ
    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("loss")
        plt.show()


# TODO 下記はコピペ
def remove_duplicate(params, grads):
    """
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    """
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif (
                    params[i].ndim == 2
                    and params[j].ndim == 2
                    and params[i].T.shape == params[j].shape
                    and np.all(params[i].T == params[j])
                ):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads
