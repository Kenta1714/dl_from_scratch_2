# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 画像を保存する用（自作）


def output_plot(filename=None, dpi=300, close=True):
    if filename is None:
        plt.show()
    else:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close()


def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    words_to_id = {}
    id_to_words = {}
    for word in words:
        if word not in words_to_id:
            # 辞書の長さをidとすることで現時点の最新index+1がnew_idとなる。
            new_id = len(words_to_id)
            words_to_id[word] = new_id
            id_to_words[new_id] = word

    # 単語列をid列に変換
    corpus = np.array([words_to_id[w] for w in words])

    return corpus, words_to_id, id_to_words


def cos_similarity(x, y, eps=1e-8):
    """_summary_

    Args:
        x (_type_): ベクトル
        y (_type_): ベクトル
        eps (_type_, optional): 0除算防止. Defaults to 1e-8.

    Returns:
        _type_: コサイン類似度。正規化してから内積を取っている。
    """
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f"{query} is not found")
        return

    print(f"\n[query]{query}")
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    # 値が大きい順にindexを取り出す。
    for i in (-1 * similarity).argsort():
        # query自身とは必ず類似度1になってしまうので除く
        if id_to_word[i] == query:
            continue
        print(f"{id_to_word[i]}: {similarity[i]}")

        count += 1
        if count >= top:
            return


def create_co_matrix(corpus, vocab_size, window_size=1):
    """共起行列を作成する

    Args:
        corpus (_type_): ID化された単語リスト
        vocab_size (_type_): 語彙数=ID数
        window_size (int, optional): 前後何単語を見るか. Defaults to 1.

    Returns:
        _type_: 共起行列
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    # idxは最初から最後までスライドするだけだが、word_idについては複数回登場する場合もあるので注意。
    for idx, word_id in enumerate(corpus):
        # 注目idxの両側にループしていく
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

        if left_idx >= 0:
            left_word_id = corpus[left_idx]
            co_matrix[word_id, left_word_id] += 1

        if right_idx < corpus_size:
            right_word_id = corpus[right_idx]
            co_matrix[word_id, right_word_id] += 1

    return co_matrix


def ppmi(C, verbose=False, eps=1e-8):
    # 行列のshapeは共起行列と同じ。
    M = np.zeros_like(C, dtype=np.float32)
    # コーパスの単語数（語彙数ではない）
    N = np.sum(C)
    # 各列で列の和をとることで、その単語が文章中にどのくらい出てくるかを求める
    S = np.sum(C, axis=0)
    # 共起行列のマス数
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            # iとjの出現回数を同じSからS[j]とS[i]のように取っており、共起行列が正方行列であり対称に近いことを前提としている。
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print(f"{100*cnt/total:.1f} done")

    return M


# TODO コピペ
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
