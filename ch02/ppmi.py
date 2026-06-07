import sys
import os
import numpy as np

sys.path.append(os.pardir)
from common.utill import preprocess, create_co_matrix, most_similar, ppmi

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)

print(f"以下の文に対するpreprocessの処理結果: {text}")
print(corpus, word_to_id, id_to_word)

C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)
print("共起行列")
print(C)
print("-" * 50)
print("PPMI")
print(W)
