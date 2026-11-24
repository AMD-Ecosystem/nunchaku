"""Kernel-level checks of the AWQ W4A16 GEMM and GEMV (used by the 4-bit T5 encoder) against torch."""

import pytest
import torch

from nunchaku._C.ops import gemm_awq, gemv_awq
from nunchaku.models.text_encoders.tinychat_utils import convert_to_tinychat_w4x16y16_linear_weight


def make_layer(n: int, k: int, group_size: int, dtype: torch.dtype, seed: int):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (n, k), generator=g).float()
    scale = (torch.rand(n, k // group_size, generator=g) * 0.01 + 0.001).to(dtype).float()
    zero = (torch.rand(n, k // group_size, generator=g) * 0.1).to(dtype).float()
    weight = (q.view(n, -1, group_size) * scale[..., None] - zero[..., None]).view(n, k)
    qweight, qscale, qzero = convert_to_tinychat_w4x16y16_linear_weight(
        weight.to(dtype), scale, zero, group_size=group_size
    )
    return weight, qweight.cuda(), qscale.cuda(), qzero.cuda()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [8, 33, 100, 150, 512])
def test_gemm_awq(m: int, dtype: torch.dtype):
    n, k, group_size = 512, 1024, 128
    weight, qweight, qscale, qzero = make_layer(n, k, group_size, dtype, seed=m)
    x = torch.randn(m, k, generator=torch.Generator().manual_seed(m + 1)).to(dtype)
    out = gemm_awq(x.cuda(), qweight, qscale, qzero).cpu().float()
    ref = x.float() @ weight.T
    err = (out - ref).abs().max() / ref.abs().max()
    assert err < (1e-2 if dtype == torch.float16 else 3e-2), f"relative error {err:.4g}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 4, 7])
def test_gemv_awq(m: int, dtype: torch.dtype):
    n, k, group_size = 512, 1024, 64  # the GEMV kernel supports group size 64 only
    weight, qweight, qscale, qzero = make_layer(n, k, group_size, dtype, seed=m * 10 + group_size)
    x = torch.randn(m, k, generator=torch.Generator().manual_seed(m + 2)).to(dtype)
    out = gemv_awq(x.cuda(), qweight, qscale, qzero, m, n, k, group_size).cpu().float()
    ref = x.float() @ weight.T
    err = (out - ref).abs().max() / ref.abs().max()
    assert err < (1e-2 if dtype == torch.float16 else 3e-2), f"relative error {err:.4g}"
