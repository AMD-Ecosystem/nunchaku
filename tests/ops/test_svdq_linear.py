"""End-to-end check of one SVDQuant W4A4 linear layer (activation quantization with the fused LoRA down
projection, the INT4 GEMM and the LoRA up projection) against the same layer computed in float."""

import pytest
import torch

from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.models.linear import SVDQW4A4Linear


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,in_features,out_features", [(100, 1024, 512), (1000, 3072, 3072)])
def test_svdq_w4a4_linear(tokens: int, in_features: int, out_features: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(tokens + in_features)
    rank, group_size = 32, 64
    qweight = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int32, generator=g)
    wscales = (torch.rand(out_features, in_features // group_size, generator=g) * 0.01 + 0.002).to(dtype)
    down = (torch.randn(rank, in_features, generator=g) * 0.05).to(dtype)
    up = (torch.randn(out_features, rank, generator=g) * 0.05).to(dtype)
    bias = (torch.randn(out_features, generator=g) * 0.1).to(dtype)
    smooth = (torch.rand(in_features, generator=g) + 0.5).to(dtype)

    packer = NunchakuWeightPacker(bits=4)
    layer = SVDQW4A4Linear(in_features, out_features, rank=rank, torch_dtype=dtype, device="cuda")
    layer.qweight.data.copy_(packer.pack_weight(qweight))
    layer.wscales.data.copy_(packer.pack_scale(wscales, group_size=group_size))
    layer.proj_down.data.copy_(packer.pack_lowrank_weight(down, down=True))
    layer.proj_up.data.copy_(packer.pack_lowrank_weight(up, down=False))
    layer.bias.data.copy_(packer.pack_scale(bias, group_size=-1))
    layer.smooth_factor.data.copy_(packer.pack_scale(smooth, group_size=-1))

    x = torch.randn(1, tokens, in_features, generator=g).to(dtype)
    out = layer(x.cuda()).cpu().float().view(tokens, out_features)

    # the kernel quantizes x / smooth, so the effective weight is W * smooth
    weight = (qweight.float().view(out_features, -1, group_size) * wscales.float()[..., None]).view(
        out_features, in_features
    )
    xf = x.float().view(tokens, in_features)
    # activation quantization as the kernel does it: per row and group of 64, scale = max|v| / 7 in 16 bits
    xs = (x.view(tokens, in_features) / smooth).float().view(tokens, -1, group_size)
    scale = (xs.abs().amax(dim=-1, keepdim=True).to(dtype).float() / 7).to(dtype).float()
    xq = (xs / scale).nan_to_num().round().clamp(-8, 7) * scale
    ref = xq.view(tokens, in_features) @ weight.T + (xf @ down.float().T) @ up.float().T + bias.float()

    rel = (out - ref).norm() / ref.norm()
    # the kernel accumulates the scaled group sums in 16 bits
    assert rel < (0.01 if dtype == torch.float16 else 0.04), f"relative error {rel:.4f}"
