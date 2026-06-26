import sys
import os
import numpy as np
import collections

sys.path.append(os.pardir)
from common.layers import Embedding, SigmoidWithLoss


class EmbeddingDot:
    """forwardでは、文脈ベクトルhと正例indexのリストを受け取り、正解の単語に対応するベクトルを抜き出してhとの内積を計算して出力する。
    """
    def __init__(self, W):
        # W_outのEmbeddingレイヤを持つ。
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        """_summary_

        Args:
            h (_type_): 周辺語の埋め込みを平均した文脈ベクトル
            idx (_type_): 正解単語のid

        Returns:
            _type_: _description_
        """
        target_W = self.embed.forward(idx)
        # axis=1のsumで要素ごとの積に対して行ごとに和をとるので、内積になる
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        # forwardのoutは一本に潰されているので二次元配列にする。
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


# TODO 下記はコピペ
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(
                    self.vocab_size, size=self.sample_size, replace=False, p=p
                )
        else:
            # GPU(cupy）で計算するときは、速度を優先
            # 負例にターゲットが含まれるケースがある
            negative_sample = np.random.choice(
                self.vocab_size,
                size=(batch_size, self.sample_size),
                replace=True,
                p=self.word_p,
            )

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        # 負例サンプリング数+正解一つ分のsigmoidとembed dotを作成する。（ここでは正解が[0]としているらしい）
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot()W for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        # sigmoidは学習可能なパラを持たないのでここに入れない。
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        """概念的には
        バッチ0:
        正例 + 負例1 + 負例2 + ...

        バッチ1:
        正例 + 負例1 + 負例2 + ...

        バッチ2:
        正例 + 負例1 + 負例2 + ...
        
        だが、実装上は下記のようになっている。なお、それぞれのloss計算でsigmoidwithloss中のcross_entropy_errorでバッチサイズ
        で割られているので、分配法則的に全部のlossを合計してからバッチサイズで割っている（つまりバッチ方向に平均）のと等価
        全バッチについて正例の loss を計算

        全バッチについて負例1個目の loss を計算
        全バッチについて負例2個目の loss を計算
        全バッチについて負例3個目の loss を計算
        ...

        Args:
            h (_type_): _description_
            target (_type_): 正例のindexのリスト

        Returns:
            _type_: _description_
        """
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例のforward
        score = self.embed_dot_layers[0].forward(h, target)
        # 正解ラベルは全て1
        correct_label = np.ones(batch_size, dtype=np.int32)
        # 全バッチついて正例のlossを計算
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例のforward
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            # 全バッチのi番目を取り出してlossを計算している
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            
            # 正例の分のlossに負例の分のlossも足し合わせる
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss
    
    def backward(self,dout=1):
        dh=0
        for l0,l1 in zip(self.loss_layers,self.embed_dot_layers):
            # SigmoidWithLossのbackward
            dscore=l0.backward(dout)
            
            # EmbeddingDotのbackward
            # dhは、正解+負例サンプリングの1セットごとに勾配を加算してセット数分（バッチサイズ分）のリストにしたやつのイメージ
            dh+=l1.backward(dscore)
        return dh
