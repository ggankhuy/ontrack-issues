import torch
from xformers.ops import fmha
import time
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    q_seqlen = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1500, 1500]
    kv_seqlen = [1530, 1530, 1506, 1529, 1528, 1528, 1527, 1527, 1526, 1526, 1525, 1525, 1524, 1501, 1523, 1523, 1522, 1522, 1521, 1521, 1520, 1520, 1519, 1519, 1518, 1518, 1517, 1517, 1516, 1516, 1515, 1515, 1514, 1514, 1513, 1513, 1512, 1512, 1511, 1511, 1510, 1510, 1509, 1509, 1508, 1508, 1507, 1507, 1506, 1505, 1505, 1504, 1504, 1503, 1503, 1502, 1502, 1501, 1500, 1500]
    options = {
        'device': 'cuda',
        'dtype': torch.bfloat16
    }
    q_shape = [1, 3058, 8, 128]
    kv_shape = [1, 491520, 8, 128]
    q = torch.randn(q_shape, **options)
    k = torch.randn(kv_shape, **options)
    v = torch.randn(kv_shape, **options)

             
    bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(        q_seqlen=q_seqlen,  kv_padding=8192,  kv_seqlen=kv_seqlen,
    )
    #t1=time.time_ns()

    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):

            y0 = fmha.memory_efficient_attention_forward(
                q,
                k,
                v,
                attn_bias=bias,
                op=None,
            )
    #t2=time.time_ns()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    #print("time: ", (t2-t1)/1000, " ms.")

if __name__ == "__main__":
    main()
