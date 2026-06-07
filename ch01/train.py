import sys
import os

sys.path.append(os.pardir)
from dataset import spiral
from common.optimizer import SGD
from common.trainer import Trainer
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

# 上記までは同じ

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()
