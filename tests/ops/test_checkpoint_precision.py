"""The precision the model loaders choose for a quantized checkpoint, on faked GPUs of every supported kind.

No GPU is used: the device capability (CUDA) or architecture (ROCm) is replaced, so this runs on any backend.
"""

import json
import types

import pytest
import torch

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.transformers import transformer_flux
from nunchaku.utils import get_checkpoint_precision, get_precision


def quantization_config(precision: str) -> dict:
    """The quantization_config a checkpoint carries in its metadata (as written by merge_safetensors)."""
    fp4 = precision == "fp4"
    return {
        "method": "svdquant",
        "weight": {
            "dtype": "fp4_e2m1_all" if fp4 else "int4",
            "scale_dtype": [None, "fp8_e4m3_nan"] if fp4 else None,
            "group_size": 16 if fp4 else 64,
        },
        "activation": {
            "dtype": "fp4_e2m1_all" if fp4 else "int4",
            "scale_dtype": "fp8_e4m3_nan" if fp4 else None,
            "group_size": 16 if fp4 else 64,
        },
        "rank": 32,
    }


CHECKPOINTS = {"int4": quantization_config("int4"), "fp4": quantization_config("fp4"), "none": {}}

# device -> precision chosen for an int4 checkpoint, an fp4 checkpoint and a checkpoint without quantization_config
EXPECTED = {
    ("cuda", (7, 5)): ("int4", "int4", "int4"),
    ("cuda", (8, 0)): ("int4", "int4", "int4"),
    ("cuda", (8, 6)): ("int4", "int4", "int4"),
    ("cuda", (8, 9)): ("int4", "int4", "int4"),
    ("cuda", (9, 0)): ("int4", "int4", "int4"),
    ("cuda", (10, 0)): ("int4", "int4", "int4"),
    ("cuda", (12, 0)): ("fp4", "fp4", "fp4"),
    ("cuda", (12, 1)): ("fp4", "fp4", "fp4"),
    ("rocm", "gfx90a"): ("int4", "int4", "int4"),
    ("rocm", "gfx942"): ("int4", "int4", "int4"),
    ("rocm", "gfx950"): ("int4", "fp4", "int4"),
    ("rocm", "gfx1100"): ("int4", "int4", "int4"),
    ("rocm", "gfx1201"): ("int4", "int4", "int4"),
    ("rocm", "gfx1250"): ("int4", "fp4", "int4"),
}


@pytest.fixture
def fake_device(monkeypatch):
    def fake(kind, device):
        if kind == "cuda":
            monkeypatch.setattr(torch.version, "hip", None)
            monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: device)

            def no_properties(*args, **kwargs):
                raise AssertionError("a CUDA device must not be asked for its ROCm architecture")

            monkeypatch.setattr(torch.cuda, "get_device_properties", no_properties)
        else:
            monkeypatch.setattr(torch.version, "hip", "7.0")
            monkeypatch.setattr(
                torch.cuda,
                "get_device_properties",
                lambda *args, **kwargs: types.SimpleNamespace(gcnArchName=f"{device}:sramecc+:xnack-"),
            )

    return fake


@pytest.mark.parametrize("kind,device", list(EXPECTED))
def test_checkpoint_precision(fake_device, kind, device):
    fake_device(kind, device)
    got = tuple(get_checkpoint_precision(CHECKPOINTS[name]) for name in ("int4", "fp4", "none"))
    assert got == EXPECTED[kind, device]


@pytest.mark.parametrize("kind,device", [key for key in EXPECTED if key[0] == "cuda"])
def test_cuda_unchanged(fake_device, kind, device):
    """On NVIDIA GPUs every checkpoint loads with the precision get_precision() chose before."""
    fake_device(kind, device)
    for config in CHECKPOINTS.values():
        assert get_checkpoint_precision(config) == get_precision()


def test_unknown_dtype_falls_back(fake_device):
    fake_device("rocm", "gfx950")
    config = quantization_config("int4")
    config["weight"]["dtype"] = "int8"
    assert get_checkpoint_precision(config) == get_precision() == "int4"


# device -> precision NunchakuFluxTransformer2dModel builds an int4 and an fp4 single-file checkpoint with
# (None: rejected by check_hardware_compatibility)
FLUX_EXPECTED = {
    ("cuda", (7, 5)): ("int4", None),
    ("cuda", (8, 0)): ("int4", None),
    ("cuda", (8, 6)): ("int4", None),
    ("cuda", (8, 9)): ("int4", None),
    ("cuda", (9, 0)): (None, None),
    ("cuda", (10, 0)): (None, None),
    ("cuda", (12, 0)): (None, "fp4"),
    ("cuda", (12, 1)): (None, "fp4"),
    ("rocm", "gfx90a"): ("int4", None),
    ("rocm", "gfx942"): ("int4", None),
    ("rocm", "gfx950"): ("int4", "fp4"),
    ("rocm", "gfx1100"): ("int4", None),
    ("rocm", "gfx1201"): ("int4", None),
    ("rocm", "gfx1250"): ("int4", "fp4"),
}


class Built(Exception):
    """Stops NunchakuFluxTransformer2dModel.from_pretrained where it builds the quantized model."""


@pytest.mark.parametrize("kind,device", list(FLUX_EXPECTED))
def test_flux_loader_precision(fake_device, monkeypatch, kind, device):
    """The FLUX loader takes a supported checkpoint's precision; on NVIDIA GPUs that is always get_precision()'s."""
    fake_device(kind, device)

    def build_model(cls, path, **kwargs):
        return types.SimpleNamespace(), {}, {"quantization_config": json.dumps(CHECKPOINTS[path.stem])}

    def load_quantized_module(*args, use_fp4, **kwargs):
        raise Built("fp4" if use_fp4 else "int4")

    monkeypatch.setattr(NunchakuFluxTransformer2dModel, "_build_model", classmethod(build_model))
    monkeypatch.setattr(transformer_flux, "load_quantized_module", load_quantized_module)
    got = []
    for name in ("int4", "fp4"):
        try:
            NunchakuFluxTransformer2dModel.from_pretrained(f"{name}.safetensors")
        except Built as built:
            got.append(str(built))
        except ValueError:
            got.append(None)
    assert tuple(got) == FLUX_EXPECTED[kind, device]
    if kind == "cuda":
        assert all(precision in (None, get_precision()) for precision in got)
