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


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


# TODO コピペ
def convert_one_hot(corpus, vocab_size):
    """one-hot表現への変換

    :param corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元もしくは3次元のNumPy配列）
    """
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


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


def to_cpu(x):
    import numpy

    # すでにnumpy配列ならそのまま
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_
        c (_type_): _description_
        word_to_id (_type_): _description_
        id_to_word (_type_): _description_
        word_matrix (_type_): word_vecs行列
        top (int, optional): _description_. Defaults to 5.
        answer (_type_, optional): _description_. Defaults to None.
    """
    # 辞書に登録されていなかった場合は何もせずに返す
    for word in (a, b, c):
        if word not in word_to_id:
            print(f"{word} is not found")
            return
    # 例えばking:man=queen:?のイメージ。この場合a-c(king-queen)とb-?(man-?)が近ければ良い。つまり?=b-a+c
    print(f"\n[analogy] {a} : {b} = {c} : ?")
    a_vec, b_vec, c_vec = (
        word_matrix[word_to_id[a]],
        word_matrix[word_to_id[b]],
        word_matrix[word_to_id[c]],
    )

    query_vec = b_vec - a_vec + c_vec
    # 理想的にはコサイン類似度になるように埋め込みベクトル側も正規化した方がいいがqueryのみ正規化している？
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    # 答えを知っている場合、答えのベクトルとqueryの内積（類似度）を返す
    if answer is not None:
        print(f"==>{answer}:{np.dot(word_matrix[word_to_id[answer]],query_vec)}")

    count = 0
    # 昇順ソートしてtop分表示
    for i in (-1 * similarity).argsort():
        # 0ベクトルなどは正規化した際にnanになりうる
        if np.isnan(similarity[i]):
            continue
        # 入力単語そのものは除外する
        if id_to_word[i] in (a, b, c):
            continue
        print(f"{id_to_word[i]}: {similarity[i]}")

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        # 一本の列ベクトルに潰すことで行ごとのベクトルのノルムを計算
        s = np.sqrt((x * x).sum(1))
        # 行ごとにノルムで割る
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        # 単純にベクトルをノルムで割る
        s = np.sqrt((x * x).sum())
        x /= s
    return x
