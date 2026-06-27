import sys
import os
import pickle
import numpy as np

sys.path.append(os.pardir)
from common.utill import most_similar, analogy

# pkl_file = "cbow_params.pkl"
pkl_file = "cbow_params_book.pkl"

with open(pkl_file, "rb") as f:
    params = pickle.load(f)
    word_vecs = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]

querys = ["you", "year", "car", "toyota"]

# most similarを当てるタスク
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# 類推タスク
print("-" * 50)
analogy("king", "man", "queen", word_to_id, id_to_word, word_vecs)
analogy("take", "took", "go", word_to_id, id_to_word, word_vecs)
analogy("car", "cars", "child", word_to_id, id_to_word, word_vecs)
analogy("good", "better", "bad", word_to_id, id_to_word, word_vecs)
