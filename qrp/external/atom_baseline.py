"""Atom uniform baseline adapter.

Loads ``quant.py`` from the vendored Atom tree (with a synthetic ``bitsandbytes``
stub, since that import is only exercised by FP4 quantization which the adapter
does not use) and exposes ``apply_atom_uniform`` — a device-agnostic uniform
group-quantisation loop over all ``nn.Linear`` layers.

Canonical settings (from ``scripts/run_atom_ppl.sh``):
    wbits  = 4,  weight_group_size = 128,  w_sym = True
    (Atom also supports abits, activation group size, outlier keeper, and
     reorder — all of which are disabled for the uniform baseline.)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from qrp.model_mapper import get_model_layers

ATOM_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "atom")
)

_vendored: Any = None


def _load_vendored():
    """Load ``model/quant.py`` via a synthetic ``atom`` package, stubbing
    ``bitsandbytes`` so the module-level import does not fail."""
    global _vendored
    if _vendored is not None:
        return _vendored

    # --- stub bitsandbytes (only quantize_fp4 / dequantize_fp4 are imported) -
    if "bitsandbytes" not in sys.modules:
        bnb = types.ModuleType("bitsandbytes")
        bnb.__path__ = []  # make it a package so sub-imports work
        bnb_func = types.ModuleType("bitsandbytes.functional")
        bnb_func.quantize_fp4 = lambda *a, **k: (None, None)
        bnb_func.dequantize_fp4 = lambda *a, **k: None
        bnb.functional = bnb_func
        sys.modules["bitsandbytes"] = bnb
        sys.modules["bitsandbytes.functional"] = bnb_func

    quant_path = Path(ATOM_DIR) / "quant.py"
    pkg = types.ModuleType("atom_model")
    pkg.__path__ = [str(quant_path.parent)]
    sys.modules.setdefault("atom_model", pkg)
    spec = importlib.util.spec_from_file_location("atom_model.quant", str(quant_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atom_model.quant"] = mod
    setattr(pkg, "quant", mod)
    spec.loader.exec_module(mod)
    _vendored = mod
    return mod


@torch.no_grad()
def apply_atom_uniform(
    model,
    calib_batches: List[torch.Tensor],
    wbits: int = 4,
    weight_group_size: int = 128,
    w_sym: bool = True,
    w_clip_ratio: float = 1.0,
    quant_type: str = "int",
    verbose: bool = True,
) -> dict:
    """Apply Atom-style uniform weight quantization to every ``nn.Linear``.

    Unlike the full Atom pipeline (mixed-precision, outlier keeper, reorder,
    activation quantisation), this adapter performs plain symmetric or
    asymmetric group quantisation with a single global bit-width — making it
    a fair uniform baseline comparable to the other external baselines.

    Parameters
    ----------
    model : nn.Module
        HuggingFace causal-LM (loaded in fp16/bf16/fp32).
    calib_batches : list[Tensor]
        Pre-tokenised calibration batches (unused here, accepted for interface
        consistency with the other adapters).
    wbits : int
        Weight bit-width (Atom paper primarily uses 4).
    weight_group_size : int
        Quantisation group size.  0 → per-channel; 128 is the Atom default.
    w_sym : bool
        Symmetric quantisation when ``True``.
    w_clip_ratio : float
        Clip ratio for weight range (1.0 = no clipping).
    quant_type : str
        ``"int"`` for uniform INT or ``"fp"`` for FP4 via bitsandbytes.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Empty – no per-layer measured statistics needed for uniform quantisation.
    """
    quant = _load_vendored()
    quantize_fn = quant.quantize_tensor_channel_group

    layers = []
    try:
        layers = get_model_layers(model)
    except AttributeError:
        pass

    n_layers = len(layers) if layers else 0
    if verbose:
        print(f"[Atom] Quantizing {n_layers} layers -> {wbits}-bit "
              f"{'symmetric' if w_sym else 'asymmetric'} group "
              f"(group_size={weight_group_size}, quant_type={quant_type}) ...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data.clone().contiguous()
            if quant_type == "int" and w.device.type == "cpu":
                # quantize_tensor_channel_group asserts contiguity but works on CPU
                pass
            module.weight.data = quantize_fn(
                w,
                n_bits=wbits,
                exponential=False,
                sym=w_sym,
                group_size=weight_group_size,
                channel_group=1,
                clip_ratio=w_clip_ratio,
                tiling=0,
                quant_type=quant_type,
            )

    if verbose:
        print("[Atom] Done.")

    return {}
