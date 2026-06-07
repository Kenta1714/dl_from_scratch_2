import sys
import os

sys.path.append(os.pardir)
from common.utill import preprocess, create_co_matrix, cos_similarity

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)

print(f"以下の文に対するpreprocessの処理結果: {text}")
print(corpus, word_to_id, id_to_word)

C = create_co_matrix(corpus, vocab_size)

print(C)

c0 = C[word_to_id["you"]]
c1 = C[word_to_id["i"]]
print(cos_similarity(c0, c1))
