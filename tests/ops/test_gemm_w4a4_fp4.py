"""Kernel-level checks of the NVFP4 W4A4 GEMM and activation quantization against a float64 reference.

NVFP4 stores 4-bit e2m1 values with one e4m3 scale per 16 values along K. The weights are packed with the
packer the checkpoints use; the activation layout is the one the quantization kernel writes.
"""

import pytest
import torch

from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from nunchaku.utils import ROCM_FP4_ARCHS, get_rocm_arch

GROUP_SIZE = 16
E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def fp4_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = get_rocm_arch()
    if arch is not None:
        return arch in ROCM_FP4_ARCHS
    return torch.cuda.get_device_capability(0) in [(12, 0), (12, 1)]


pytestmark = pytest.mark.skipif(not fp4_supported(), reason="the GPU has no NVFP4 kernels")


def act_layout(m: int, k: int):
    """Row, k and nibble shift of every 4-bit activation value in the packed layout, as flat uint32 indices.

    Blocks of 256 rows: 8 warps of 32 rows, two 16-row tiles per warp. Per tile and 64 values of K, lane 4g + t
    holds four words: (row g, k 8t..8t+7), (row g + 8, same k), (row g, k 32+8t..), (row g + 8, k 32+8t..).
    """
    blk, kstep, warp, tile, g, t, word, e = torch.meshgrid(
        *[torch.arange(n) for n in (m // 256, k // 64, 8, 2, 8, 4, 4, 8)], indexing="ij"
    )
    row = blk * 256 + warp * 32 + tile * 16 + g + 8 * (word % 2)
    col = kstep * 64 + 32 * (word // 2) + 8 * t + e
    index = ((((blk * (k // 64) + kstep) * 8 + warp) * 2 + tile) * 32 + 4 * g + t) * 4 + word
    return row.flatten(), col.flatten(), index.flatten(), (4 * e).flatten()


def ascale_layout(m: int, k: int):
    """Row, group and byte index of every activation scale in the packed layout.

    Per block, 64 values of K and warp, lane 4g + 2i + h holds the four group scales of row 16i + g + 8h.
    """
    blk, kstep, warp, g, i, h, b = torch.meshgrid(
        *[torch.arange(n) for n in (m // 256, k // 64, 8, 8, 2, 2, 4)], indexing="ij"
    )
    row = blk * 256 + warp * 32 + 16 * i + g + 8 * h
    group = kstep * 4 + b
    index = ((((blk * (k // 64) + kstep) * 8 + warp) * 32 + 4 * g + 2 * i + h) * 4) + b
    return row.flatten(), group.flatten(), index.flatten()


def pack_act_fp4(codes: torch.Tensor) -> torch.Tensor:
    m, k = codes.shape
    row, col, index, shift = act_layout(m, k)
    words = torch.zeros(m * k // 8, dtype=torch.int64)
    words.index_add_(0, index, codes[row, col].long() << shift)
    return words.to(torch.int32).view(torch.uint8).view(m, k // 2)


def unpack_act_fp4(packed: torch.Tensor, m: int, k: int) -> torch.Tensor:
    row, col, index, shift = act_layout(m, k)
    words = packed.contiguous().view(torch.int32).flatten().long() & 0xFFFFFFFF
    codes = torch.empty(m, k, dtype=torch.long)
    codes[row, col] = (words[index] >> shift) & 0xF
    return codes


def pack_ascales_fp4(scales: torch.Tensor) -> torch.Tensor:
    """scales: (M, K / 16) float8_e4m3fn -> the (K / 16, M) float8 tensor the GEMM reads."""
    m, groups = scales.shape
    row, group, index = ascale_layout(m, groups * GROUP_SIZE)
    out = torch.empty(m * groups, dtype=torch.uint8)
    out[index] = scales.view(torch.uint8)[row, group]
    return out.view(torch.float8_e4m3fn).view(groups, m)


def unpack_ascales_fp4(packed: torch.Tensor, m: int, k: int) -> torch.Tensor:
    row, group, index = ascale_layout(m, k)
    out = torch.empty(m, k // GROUP_SIZE, dtype=torch.float64)
    out[row, group] = packed.contiguous().view(torch.uint8).flatten()[index].view(torch.float8_e4m3fn).double()
    return out


def random_scales(shape, generator, extremes=(2.0**-9, 0.0)) -> torch.Tensor:
    """Random e4m3 scales from 2^-6 to 4, with the given extreme values in the first groups of the first row."""
    s = torch.exp2(torch.rand(shape, generator=generator) * 8 - 6).to(torch.float8_e4m3fn)
    s[0, : len(extremes)] = torch.tensor(extremes).to(torch.float8_e4m3fn)
    return s


def dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = E2M1_VALUES[codes.long()]
    return values * scales.double().repeat_interleave(GROUP_SIZE, dim=1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,n,k", [(256, 128, 128), (512, 384, 1024), (256, 3072, 3072)])
def test_gemm_w4a4_fp4(m: int, n: int, k: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(m * 7 + n * 3 + k + (dtype == torch.bfloat16))
    act_codes = torch.randint(0, 16, (m, k), generator=g)
    wgt_codes = torch.randint(0, 16, (n, k), generator=g)
    act_codes[:2] = 7  # +6
    wgt_codes[:2] = 15  # -6
    # the largest scale only on the activation side, so that no output overflows 16 bits
    ascales = random_scales((m, k // GROUP_SIZE), g, extremes=(448.0, 2.0**-9, 0.0))
    wscales = random_scales((n, k // GROUP_SIZE), g)
    alpha = 2.0**-8
    wcscales = (torch.rand(n, generator=g) + 0.5).to(dtype)
    rank = 16

    packer = NunchakuWeightPacker(bits=4)
    out = torch.empty(m, n, dtype=dtype, device="cuda")
    svdq_gemm_w4a4_cuda(
        act=pack_act_fp4(act_codes).cuda(),
        wgt=packer.pack_weight(wgt_codes.to(torch.int32)).cuda(),
        out=out,
        ascales=pack_ascales_fp4(ascales).cuda(),
        wscales=packer.pack_scale(wscales.to(dtype), group_size=GROUP_SIZE).cuda(),
        lora_act_in=torch.zeros(m, rank, dtype=torch.float32, device="cuda"),
        lora_up=torch.zeros(n, rank, dtype=dtype, device="cuda"),
        fp4=True,
        alpha=alpha,
        wcscales=packer.pack_scale(wcscales, group_size=-1).cuda(),
    )
    torch.cuda.synchronize()

    ref = dequantize(act_codes, ascales) @ dequantize(wgt_codes, wscales).T * alpha * wcscales.double()
    err = (out.cpu().double() - ref).abs()
    # products are exact and summed in fp32; the result is rounded to 16 bits before the channel scale
    tol = 2 * 2.0**-11 if dtype == torch.float16 else 2 * 2.0**-8
    bound = tol * ref.abs().amax(dim=1, keepdim=True)
    assert (err <= bound).all(), f"max error / row max {(err / bound * tol).max().item():.4g}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,k", [(256, 256), (1000, 3072)])
def test_quantize_w4a4_act_fp4(m: int, k: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(m + k)
    rank = 32
    # row magnitudes from 2^-12 to 2^6: scales from the e4m3 subnormals (and zero) up to below saturation at 448
    x = (torch.randn(m, k, generator=g) * torch.exp2(torch.rand(m, 1, generator=g) * 18 - 12)).to(dtype)
    x[0, :16] = 0  # an all-zero group
    smooth = (torch.rand(k, generator=g) + 0.5).to(dtype)
    down = (torch.randn(rank, k, generator=g) * 0.05).to(dtype)

    packer = NunchakuWeightPacker(bits=4)
    q, s, _ = svdq_quantize_w4a4_act_fuse_lora_cuda(
        x.cuda(),
        lora_down=packer.pack_lowrank_weight(down, down=True).cuda(),
        smooth=packer.pack_scale(smooth, group_size=-1).cuda(),
        fp4=True,
    )
    torch.cuda.synchronize()
    m_pad = q.shape[0]
    codes = unpack_act_fp4(q.cpu(), m_pad, k)[:m]
    scales = unpack_ascales_fp4(s.cpu(), m_pad, k)[:m]

    v = (x / smooth).double().view(m, -1, GROUP_SIZE)
    vmax = v.abs().amax(dim=-1)
    # scale = max|v| / 6 rounded to e4m3; values divided by the unrounded scale and rounded to e2m1
    expect = (vmax / 6).clamp(max=448).to(torch.float8_e4m3fn).double()
    assert (scales - expect).abs().le(expect * 2.0**-3 + 2.0**-9).all(), "scale off by more than one e4m3 step"
    deq = dequantize(codes, scales).view(m, -1, GROUP_SIZE)
    # half the largest e2m1 step (1 of 6 at the top, of the rounded scale) plus the scale's rounding on values up
    # to 6; e4m3 rounds to 2^-4 relative, or to 2^-10 absolute below 2^-6, where its values are subnormal
    s = vmax / 6
    ds = torch.maximum(s * 2.0**-4, torch.full_like(s, 2.0**-10))
    bound = (s + 7 * ds) * 1.01 + 2.0**-20
    assert (deq - v).abs().le(bound[..., None]).all(), "quantized activations outside the rounding bound"
    rel = (deq - v).norm() / v.norm()
    assert rel < 0.12, f"relative quantization error {rel:.4f}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,in_features,out_features", [(100, 1024, 512), (1000, 3072, 3072)])
def test_svdq_w4a4_linear_nvfp4(tokens: int, in_features: int, out_features: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(tokens + in_features + 1)
    rank = 32
    wgt_codes = torch.randint(0, 16, (out_features, in_features), generator=g)
    wscales = torch.exp2(torch.rand(out_features, in_features // GROUP_SIZE, generator=g) * 4 - 2)
    wscales = wscales.to(torch.float8_e4m3fn)
    down = (torch.randn(rank, in_features, generator=g) * 0.05).to(dtype)
    up = (torch.randn(out_features, rank, generator=g) * 0.05).to(dtype)
    bias = (torch.randn(out_features, generator=g) * 0.1).to(dtype)
    smooth = (torch.rand(in_features, generator=g) + 0.5).to(dtype)
    alpha = 0.01

    packer = NunchakuWeightPacker(bits=4)
    layer = SVDQW4A4Linear(
        in_features, out_features, rank=rank, precision="nvfp4", torch_dtype=dtype, device="cuda"
    )
    layer.qweight.data.copy_(packer.pack_weight(wgt_codes.to(torch.int32)))
    layer.wscales.data.copy_(packer.pack_scale(wscales.to(dtype), group_size=GROUP_SIZE))
    layer.proj_down.data.copy_(packer.pack_lowrank_weight(down, down=True))
    layer.proj_up.data.copy_(packer.pack_lowrank_weight(up, down=False))
    layer.bias.data.copy_(packer.pack_scale(bias, group_size=-1))
    layer.smooth_factor.data.copy_(packer.pack_scale(smooth, group_size=-1))
    layer.wtscale = alpha

    x = torch.randn(1, tokens, in_features, generator=g).to(dtype)
    out = layer(x.cuda()).cpu().double().view(tokens, out_features)

    # the kernel quantizes x / smooth to NVFP4, so the effective weight is W * smooth; fp4 quantization of the
    # activations is simulated in float64 with the scale rounded to e4m3 as the kernel rounds it
    weight = dequantize(wgt_codes, wscales) * alpha
    xf = x.double().view(tokens, in_features)
    v = (x.view(tokens, in_features) / smooth).double().view(tokens, -1, GROUP_SIZE)
    scale = (v.abs().amax(dim=-1, keepdim=True) / 6).to(torch.float8_e4m3fn).double()
    grid = E2M1_VALUES[:8]
    r = (v.abs() / (v.abs().amax(dim=-1, keepdim=True) / 6)).nan_to_num()
    xq = grid[(r[..., None] - grid).abs().argmin(dim=-1)] * v.sign() * scale
    ref = xq.view(tokens, in_features) @ weight.T + (xf @ down.double().T) @ up.double().T + bias.double()

    rel = (out - ref).norm() / ref.norm()
    # ties and e2m1 rounding of values near a midpoint may go either way
    assert rel < (0.01 if dtype == torch.float16 else 0.03), f"relative error {rel:.4f}"
