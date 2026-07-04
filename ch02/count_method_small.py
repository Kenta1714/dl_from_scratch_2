import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from common.utill import preprocess, create_co_matrix, ppmi, output_plot

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)

print(f"以下の文に対するpreprocessの処理結果: {text}")
print(corpus, word_to_id, id_to_word)

C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

U, S, V = np.linalg.svd(W)

np.set_printoptions(precision=3)
print(C)
print(W)
print(U)

for word, word_id in word_to_id.items():
    # 第二引数の地点に第一引数のアノテーションをつける関数
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
# Uの２列目までを見ることで、各単語を2次元ベクトルとして表せる。
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
output_plot(filename="単語ベクトルのSVDによる圧縮結果_")
