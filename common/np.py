from common.config import GPU

# ソースコードが全てnpで記述されているので、GPUモードの場合はcupyをnpとしてimportすることでcupyの関数やメソッドとして動かす
if GPU:
    import cupy as np

    np.cuda.set_allo
