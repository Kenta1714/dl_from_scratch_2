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
