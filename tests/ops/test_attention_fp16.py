"""Kernel-level check of the nunchaku-fp16 attention kernel against torch SDPA.

Q, K and V are packed here in the layouts the QKV-projection epilogue writes for this kernel: 16x16 tiles in
the per-lane mma.sync fragment order, Q as the A operand, K and V as B operands (V transposed per 8x8 block).
"""

import pytest
import torch

from nunchaku._C.ops import attention_fp16

HEAD_DIM = 128


def lane_tiles(x: torch.Tensor) -> torch.Tensor:
    """(..., 16, 16) tiles -> (..., 32 lanes, 4 registers, 2 halves) in A-fragment order.

    Register r of lane 4g + t holds row g + 8 * (r % 2), columns 8 * (r // 2) + 2t and + 1.
    """
    t = x.unflatten(-2, (2, 8)).unflatten(-1, (2, 4, 2))  # (..., row hi, g, col hi, t, pair)
    t = t.permute(*range(t.dim() - 5), -4, -2, -3, -5, -1)  # (..., g, t, col hi, row hi, pair)
    return t.flatten(-5, -4).flatten(-3, -2)


def pack_q(q: torch.Tensor) -> torch.Tensor:
    # [B, H, T, D] -> [B, H, T / 16, D / 16, lanes, 4, 2]: registers (g, lo), (g + 8, lo), (g, hi), (g + 8, hi)
    b, h, t, d = q.shape
    tiles = q.view(b, h, t // 16, 16, d // 16, 16).transpose(3, 4)
    return lane_tiles(tiles).contiguous().view(b, h, t, d)


def pack_k(k: torch.Tensor) -> torch.Tensor:
    # B operand: registers (g, lo), (g, hi), (g + 8, lo), (g + 8, hi) with g a token
    b, h, t, d = k.shape
    tiles = lane_tiles(k.view(b, h, t // 16, 16, d // 16, 16).transpose(3, 4))
    return tiles[..., [0, 2, 1, 3], :].contiguous().view(b, h, t, d)


def pack_v(v: torch.Tensor) -> torch.Tensor:
    # [B, H, T, D] -> per (16 tokens, 16 dims) tile, the A-fragment order of the transposed tile with each
    # 8x8 block's position kept: registers (dim g, tok lo), (dim g, tok hi), (dim g + 8, tok lo), (dim g + 8, tok hi)
    b, h, t, d = v.shape
    tiles = v.view(b, h, t // 16, 16, d // 16, 16).transpose(3, 4)  # (..., tok tile, dim tile, tok, dim)
    return lane_tiles(tiles.transpose(-1, -2))[..., [0, 2, 1, 3], :].contiguous().view(b, h, t, d)


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,valid", [(256, 256), (1024, 1024), (512, 500)])
def test_attention_fp16(tokens: int, valid: int, out_dtype: torch.dtype):
    g = torch.Generator().manual_seed(tokens + valid)
    b, h = 2, 3
    q, k, v = (torch.randn(b, h, tokens, HEAD_DIM, generator=g).half() for _ in range(3))
    # padding tokens: the QKV epilogue writes NaN keys and zero values for them
    k[:, :, valid:] = float("nan")
    v[:, :, valid:] = 0
    scale = HEAD_DIM**-0.5
    o = torch.empty(b, tokens, h * HEAD_DIM, dtype=out_dtype, device="cuda")
    attention_fp16(pack_q(q).cuda(), pack_k(k).cuda(), pack_v(v).cuda(), o, scale)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k[:, :, :valid].float(), v[:, :, :valid].float(), scale=scale
    )
    ref = ref.transpose(1, 2).reshape(b, tokens, h * HEAD_DIM)
    err = (o.cpu().float() - ref).abs().max()
    assert torch.isfinite(o).all()
    assert err < (2e-2 if out_dtype == torch.float16 else 5e-2), f"max error {err:.4g}"
