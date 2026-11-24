"""Kernel-level checks of the W4A4 GEMM against a float64 reference.

The weights and scales are packed with the same packer the LoRA converter uses, so these tests also check
that the kernels read the packed layout the checkpoints are stored in.
"""

import pytest
import torch

from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda

GROUP_SIZE = 64


def pack_act(qact: torch.Tensor) -> torch.Tensor:
    """Pack int4 activations of shape (M, K) into the layout the GEMM reads."""
    m, k = qact.shape
    packed = NunchakuWeightPacker(bits=4, warp_n=256).pack_weight(qact)
    # Each lane holds four 32-bit registers. The weight packer orders them (row r, k lo), (row r, k hi),
    # (row r + 8, k lo), (row r + 8, k hi); the activation fragment is (r, k lo), (r + 8, k lo), (r, k hi),
    # (r + 8, k hi).
    packed = packed.view(-1, 4, 4)[:, [0, 2, 1, 3], :]
    return packed.reshape(m, k // 2).contiguous()


def pack_ascales(ascales: torch.Tensor) -> torch.Tensor:
    """Pack activation scales of shape (M, K / 64) into the layout the GEMM reads."""
    m, groups = ascales.shape
    # Per block of 256 rows: 8 warps of 32 rows; lane l of a warp holds rows 16 * (l // 8) + l % 8 and that + 8.
    s = ascales.view(m // 256, 8, 2, 2, 8, groups)  # (block, warp, l // 8, +8, l % 8, group)
    s = s.permute(0, 5, 1, 2, 4, 3).contiguous()  # (block, group, warp, l // 8, l % 8, +8)
    return s.view(groups, m)


def reference(qact, ascales, qwgt, wscales):
    m, k = qact.shape
    out = torch.zeros(m, qwgt.shape[0], dtype=torch.float64)
    for g in range(k // GROUP_SIZE):
        cols = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
        psum = qact[:, cols].double() @ qwgt[:, cols].double().T
        out += psum * ascales[:, g : g + 1].double() * wscales[:, g].double()[None, :]
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("act_unsigned", [False, True])
@pytest.mark.parametrize("m,n,k", [(256, 128, 128), (512, 384, 1024), (256, 3072, 3072)])
def test_gemm_w4a4_int4(m: int, n: int, k: int, act_unsigned: bool, dtype: torch.dtype):
    torch.manual_seed(m * 7 + n * 3 + k + int(act_unsigned))
    if act_unsigned:
        qact = torch.randint(0, 16, (m, k), dtype=torch.int32)
    else:
        qact = torch.randint(-8, 8, (m, k), dtype=torch.int32)
    qwgt = torch.randint(-8, 8, (n, k), dtype=torch.int32)
    ascales = (torch.rand(m, k // GROUP_SIZE) * 0.02 + 0.001).to(dtype)
    wscales = (torch.rand(n, k // GROUP_SIZE) * 0.02 + 0.001).to(dtype)
    rank = 16

    packer = NunchakuWeightPacker(bits=4)
    out = torch.empty(m, n, dtype=dtype, device="cuda")
    svdq_gemm_w4a4_cuda(
        act=pack_act(qact).cuda(),
        wgt=packer.pack_weight(qwgt).cuda(),
        out=out,
        ascales=pack_ascales(ascales).cuda(),
        wscales=packer.pack_scale(wscales, group_size=GROUP_SIZE).cuda(),
        lora_act_in=torch.zeros(m, rank, dtype=torch.float32, device="cuda"),
        lora_up=torch.zeros(n, rank, dtype=dtype, device="cuda"),
        act_unsigned=act_unsigned,
    )
    torch.cuda.synchronize()

    ref = reference(qact, ascales, qwgt, wscales)
    err = (out.cpu().double() - ref).abs()
    scale = ref.abs().max()
    # The kernel converts each 64-wide integer partial sum to 16 bits and accumulates in 16 bits.
    tol = 4e-3 if dtype == torch.float16 else 3e-2
    assert err.max() <= tol * scale, f"max error {err.max().item():.4g} vs scale {scale.item():.4g}"
    assert err.mean() <= tol / 8 * scale
