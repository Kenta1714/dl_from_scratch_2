from common.config import GPU

# ソースコードが全てnpで記述されているので、GPUモードの場合はcupyをnpとしてimportすることでcupyの関数やメソッドとして動かす
if GPU:
    pass
    # import cupy as np

    # print()

else:
    import numpy as np
