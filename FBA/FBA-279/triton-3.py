import unittest
import torch
import triton
import triton.language as tl
import triton.testing
@triton.jit

def _rms_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    eps,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)

    ## Attention RMS NORM
    _var = float(0.0)
    # _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, D, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(
            x_ptr + D * b + cols, mask=cols < D, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        _var += tl.sum(a * a, axis=0)
    rstd = tl.math.rsqrt((_var / D) + eps)
    for offset in range(0, D, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        a = tl.load(
            x_ptr + D * b + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask)
        tl.store(y_ptr + D * b + cols, a * rstd * w, mask=mask)

def rms_norm(x, w, eps=1.0e-5):
    y = torch.empty_like(x)
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert y.is_contiguous()
    (B, T, D) = x.shape
    assert w.shape == (D,)
    assert y.shape == x.shape
    _rms_norm_kernel[(B * T,)](
        x,
        y,
        w,
        eps,
        D,
        BLOCK_SIZE=triton.next_power_of_2(D),
    )
    return y

class LLamaTests(unittest.TestCase):
    def test_rms_norm(self):
        D = 8192
        B = 2
        T = 4096
        x = torch.randn(size=(B, T, D), dtype=torch.bfloat16, device="cuda")
        w = torch.randn(size=(D,), dtype=torch.bfloat16, device="cuda")

        def ref_rms_norm(x, w):
            x_std = torch.sqrt(torch.mean(x**2, -1, keepdim=True))
            x_norm = x / (x_std + 1.0e-6)
            return w * x_norm

        torch.testing.assert_close(ref_rms_norm(x, w), rms_norm(x, w))
        t_ms = triton.testing.do_bench(lambda: rms_norm(x, w))
        t_seconds = t_ms / 1.0e3
        bandwidth_gbs = x.numel() * x.element_size() * 2 / t_seconds / 1.0e9
        print(f"T: {t_seconds * 1.0e6:.2f}us, BW: {bandwidth_gbs:.2f}GB/s")

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            rms_norm(x, w)

        for _ in range(10):
            g.replay()

if __name__ == "__main__":
    unittest.main()
