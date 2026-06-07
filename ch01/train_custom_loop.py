import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from dataset import spiral
from common.optimizer import SGD
from common import utill
from two_layer_net import TwoLayerNet

# ハイパラ
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# データ読み込み
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

data_size = len(x)
# ミニバッチを何回繰り返したら1エポックになるか
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):

    # indexをランダムに並び替える（2回目以降はランダムな状態から別のランダムな状態に並び替えられるので問題ない）
    idx = np.random.permutation(data_size)  # 0からdata_size-1までの数字がランダムに並ぶ
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
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(
                "| epoch %d | iter %d / %d | loss %.2f "
                % (epoch + 1, iters + 1, max_iters, avg_loss)
            )
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

# 以下コピペ
# 学習結果のプロット
plt.plot(np.arange(len(loss_list)), loss_list, label="train")
plt.xlabel("iterations (x10)")
plt.ylabel("loss")
utill.output_plot(filename="loss")

# 境界領域のプロット
h = 0.001
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis("off")

# データ点のプロット
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ["o", "x", "^"]
for i in range(CLS_NUM):
    plt.scatter(
        x[i * N : (i + 1) * N, 0], x[i * N : (i + 1) * N, 1], s=40, marker=markers[i]
    )
utill.output_plot(filename="決定境界")
