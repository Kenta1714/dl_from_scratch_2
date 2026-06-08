import sys
import os
import numpy as np

sys.path.append(os.pardir)
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")

print(f"corpus size: {len(corpus)}")
n = 30
print(f"corpus[:{n}]: {corpus[:n]}")
print()

print([id_to_word[id] for id in corpus[:n]])

id_list = [0, 1, 2]
for id in id_list:
    print(f"id_to_word[{id}]: {id_to_word[id]}")
print()

word_list = ["car", "happy", "lexus", "star", "wars", "toyota"]
for word in word_list:
    print(f"word_to_id[{word}]: {word_to_id[word]}")
