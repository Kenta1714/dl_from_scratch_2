import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

sys.path.append(os.pardir)
from common.utill import most_similar, create_co_matrix, ppmi, output_plot

from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
print("共起行列作成中...")
C = create_co_matrix(corpus, vocab_size, window_size)
print("PPMI行列作成中...")
W = ppmi(C, verbose=True)

print("SVD実行中...")
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota", "dog"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

if wordvec_size == 2:
    n = 100
    for i, (word, word_id) in enumerate(word_to_id.items()):
        # 第二引数の地点に第一引数のアノテーションをつける関数
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
        if i >= n:
            break
    # Uの２列目までを見ることで、各単語を2次元ベクトルとして表せる。
    plt.scatter(U[:n, 0], U[:n, 1], alpha=0.5)
    plt.show()
