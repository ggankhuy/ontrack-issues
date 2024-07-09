import torch
from xformers.ops import fmha


def main():
    xattn_q_seqlen = [3, 5, 3, 2]
    xattn_kv_seqlen0 = [0, 1028, 0, 1028]
    xattn_kv_seqlen1 = [1, 1028, 1, 1028]
    options = {
        'device': 'cuda',
        'dtype': torch.bfloat16
    }
    q_shape = (1, 13, 8, 1, 32)
    kv_shape = (1, 2056, 8, 1, 32)
    axq0 = torch.randn(q_shape, **options)
    axk0 = torch.randn(kv_shape, **options)
    axv0 = torch.randn(kv_shape, **options)

    kv_dummy = torch.randn((1, 1, 8, 1, 32), **options)

    axk1 = torch.concatenate((kv_dummy, axk0[:, :1028, ...], kv_dummy, axk0[:, 1028:, ...]), dim=1)
    axv1 = torch.concatenate((kv_dummy, axv0[:, :1028, ...], kv_dummy, axv0[:, 1028:, ...]), dim=1)
                            
    xattn_bias0 = fmha.attn_bias.BlockDiagonalMask.from_seqlens(
        q_seqlen=xattn_q_seqlen, kv_seqlen=xattn_kv_seqlen0
    )
    xattn_bias1 = fmha.attn_bias.BlockDiagonalMask.from_seqlens(
        q_seqlen=xattn_q_seqlen, kv_seqlen=xattn_kv_seqlen1
    )
    print(f"{xattn_bias0=}")
    y0 = fmha.memory_efficient_attention_forward(
        axq0,
        axk0,
        axv0,
        attn_bias=xattn_bias0,
        op=None,
    )
    y1 = fmha.memory_efficient_attention_forward(
        axq0,
        axk1,
        axv1,
        attn_bias=xattn_bias1,
        op=None,
    )
    # print(f"@@@@@@@@@ {y.shape=} {y=}")
    mask = ~y0.isnan()
    print(mask.shape)
    torch.testing.assert_close(y0[mask], y1[mask])
    # look for NaNs ^

if __name__ == "__main__":
    main()
