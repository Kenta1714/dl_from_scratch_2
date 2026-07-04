import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from common.optimizer import SGD
from common.utill import output_plot
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# ハイパラ設定
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5  # Truncated BPTTの展開する時間サイズ
lr = 0.1
max_epoch = 100

# データ読み込み
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

# 入力は最初の単語から最後の一個の手前の単語まで
xs = corpus[:-1]
# 正解ラベルは次単語なので、最初の単語を除き一つずらしたもの
ts = corpus[1:]
data_size = len(xs)

print(f"corupus size: {corpus_size}, vocabulary size: {vocab_size}")

# 何回ミニバッチを取り出すか。一回でバッチサイズ10でそれぞれ5単語ずつなので50単語。よって大体20回ミニバッチを取り出すと1エポック
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# モデルなどの作成
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# バッチサイズ分に分割して、それぞれの先頭のindexを格納したリストを作成
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # ミニバッチの取得。単語IDなのでint型の要素の空配列を作成。
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")
        # t=1~5を順番にループ
        for t in range(time_size):
            # 各tで、10分割ならそれぞれのt番目を格納する
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # 同じインスタンスなので内部のself.hがしっかりforループの間で引き継がれる（stateful=Trueなので）
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
    # エポックごとにパープレキシティの評価
    ppl = np.exp(total_loss / loss_count)
    print(f"| epoch {epoch+1} | perprlexity {ppl:.2f}")
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0


# 以下コピペ
# グラフの描画
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label="train")
plt.xlabel("epochs")
plt.ylabel("perplexity")
output_plot(filename=f"学習曲線_corupus_size_{corpus_size}_epoch_{max_epoch}")
