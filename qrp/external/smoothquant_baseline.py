"""SmoothQuant uniform baseline adapter.

Loads only ``smoothquant/fake_quant.py`` (for ``quantize_weight_per_channel_absmax``)
from the vendored SmoothQuant tree and implements the activation-smoothing +
per-channel-absmax weight quantization pipeline generically, without importing
``smoothquant.smooth`` (which has hard-coded transformer class imports).

Canonical settings (from the SmoothQuant paper / ``generate_act_scales.py``):
    - W8A8:  8-bit weights, 8-bit activations  (``wbits=8``, ``alpha=0.5``)
    - W4A8:  4-bit weights, 8-bit activations  (``wbits=4``, ``alpha=0.5``)
    - Activation scales are either pre-computed (HuggingFace repo
      ``mit-han-lab/smoothquant-scales``) or computed on-the-fly from the
      calibration data supplied by the benchmark.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from qrp.model_mapper import get_layer_structure, get_model_layers

SMOOTHQUANT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "smoothquant")
)

_vendored: Any = None


def _load_vendored():
    """Load ``smoothquant/fake_quant.py`` via synthetic package."""
    global _vendored
    if _vendored is not None:
        return _vendored
    fq_path = Path(SMOOTHQUANT_DIR) / "fake_quant.py"
    pkg = types.ModuleType("smoothquant")
    pkg.__path__ = [str(fq_path.parent)]
    sys.modules.setdefault("smoothquant", pkg)
    spec = importlib.util.spec_from_file_location(
        "smoothquant.fake_quant", str(fq_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoothquant.fake_quant"] = mod
    setattr(pkg, "fake_quant", mod)
    spec.loader.exec_module(mod)
    _vendored = mod
    return mod


def _get_decoder_layers(model):
    try:
        return get_model_layers(model)
    except AttributeError:
        return []


def _layer_norms(layer):
    """Return ``(attn_norm, ffn_norm)`` for a decoder layer."""
    if hasattr(layer, "input_layernorm") and hasattr(
        layer, "post_attention_layernorm"
    ):
        return layer.input_layernorm, layer.post_attention_layernorm
    if hasattr(layer, "self_attn_layer_norm") and hasattr(
        layer, "final_layer_norm"
    ):
        return layer.self_attn_layer_norm, layer.final_layer_norm
    if hasattr(layer, "ln_1") and hasattr(layer, "ln_2"):
        return layer.ln_1, layer.ln_2
    raise ValueError(
        f"Cannot detect norms on {type(layer).__name__}; "
        "supported patterns: LLaMA, OPT, GPT-2."
    )


def _qkv_projections(attn_module):
    """Return the Linear modules for Q, K, V (exclude o_proj / out_proj)."""
    qkv_names = ("q_proj", "k_proj", "v_proj")
    return [
        getattr(attn_module, n)
        for n in qkv_names
        if hasattr(attn_module, n)
    ]


def _ffn_fc1_projections(ffn_module):
    """Return the Linear modules that act as the FFN first layer
    (gate_proj+up_proj for LLaMA, fc1 for OPT, dense_h_to_4h for Bloom, etc.)."""
    fc1_names = ("gate_proj", "up_proj", "fc1", "dense_h_to_4h", "w1", "w3")
    return [
        getattr(ffn_module, n)
        for n in fc1_names
        if hasattr(ffn_module, n)
    ]


def _ffn_scale_key(layer_idx: int, ffn_module, fc1_name: str) -> str:
    """Infer the act_scales dict key for the first FFN layer of a given block."""
    # LLaMA / Mistral / Mixtral  →  model.layers.{i}.mlp.gate_proj
    # OPT                         →  model.layers.{i}.fc1
    # Bloom                       →  model.layers.{i}.mlp.dense_h_to_4h
    if hasattr(ffn_module, "gate_proj") and fc1_name == "gate_proj":
        return f"model.layers.{layer_idx}.mlp.gate_proj"
    if fc1_name == "fc1":
        return f"model.layers.{layer_idx}.fc1"
    if fc1_name == "dense_h_to_4h":
        return f"model.layers.{layer_idx}.mlp.dense_h_to_4h"
    # Generic fallback – assume LLaMA-style mlp path.
    return f"model.layers.{layer_idx}.mlp.{fc1_name}"


@torch.no_grad()
def _compute_act_scales(model, calib_batches: List[torch.Tensor]) -> Dict:
    """Collect per-channel max-absolute activation values over calibration data.

    Returns a dict  ``{module_name: Tensor(hidden_dim)}``  suitable for
    ``_smooth_weights``.
    """
    model.eval()
    act_scales: Dict[str, torch.Tensor] = {}
    hooks: list = []

    def _stat_hook(m, x, _y, name):
        inp = x[0] if isinstance(x, tuple) else x
        flat = inp.reshape(-1, inp.shape[-1]).abs().detach()
        chan_max = flat.max(dim=0)[0].float().cpu()
        act_scales[name] = (
            torch.max(act_scales[name], chan_max)
            if name in act_scales
            else chan_max
        )

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(_stat_hook, name=name)
                )
            )

    for batch in calib_batches:
        model(batch)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def _smooth_weights(model, act_scales: Dict, alpha: float = 0.5) -> None:
    """Apply SmoothQuant weight smoothing (in-place).

    For every decoder layer, scales are computed from the cached activation
    scales and the current weight magnitudes, then applied to the norm
    parameters and the subsequent projection weights.
    """
    layers = _get_decoder_layers(model)
    for i, layer in enumerate(layers):
        try:
            attn_norm, ffn_norm = _layer_norms(layer)
        except ValueError:
            continue

        attn_module, attn_projs = get_layer_structure(layer)[0]
        ffn_module, ffn_projs = get_layer_structure(layer)[1]

        qkvs = _qkv_projections(attn_module)
        if qkvs:
            w_scales = (
                torch.cat(
                    [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in qkvs],
                    dim=0,
                )
                .max(dim=0)[0]
                .clamp(min=1e-5)
            )
            key = f"model.layers.{i}.self_attn.q_proj"
            if key in act_scales:
                a_scales = act_scales[key].to(
                    device=w_scales.device, dtype=w_scales.dtype
                )
                scales = (
                    (a_scales.pow(alpha) / w_scales.pow(1 - alpha))
                    .clamp(min=1e-5)
                )
                if hasattr(attn_norm, "weight"):
                    attn_norm.weight.div_(scales)
                if getattr(attn_norm, "bias", None) is not None:
                    attn_norm.bias.div_(scales)
                for fc in qkvs:
                    fc.weight.mul_(scales.view(1, -1))

        fc1_modules = _ffn_fc1_projections(ffn_module)
        if fc1_modules:
            w_scales = (
                torch.cat(
                    [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fc1_modules],
                    dim=0,
                )
                .max(dim=0)[0]
                .clamp(min=1e-5)
            )
            fc1_name = next(
                n
                for n in ("gate_proj", "fc1", "dense_h_to_4h", "up_proj", "w1", "w3")
                if hasattr(ffn_module, n)
            )
            key = _ffn_scale_key(i, ffn_module, fc1_name)
            if key in act_scales:
                a_scales = act_scales[key].to(
                    device=w_scales.device, dtype=w_scales.dtype
                )
                scales = (
                    (a_scales.pow(alpha) / w_scales.pow(1 - alpha))
                    .clamp(min=1e-5)
                )
                if hasattr(ffn_norm, "weight"):
                    ffn_norm.weight.div_(scales)
                if getattr(ffn_norm, "bias", None) is not None:
                    ffn_norm.bias.div_(scales)
                for fc in fc1_modules:
                    fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def _quantize_all_linear(model, wbits: int) -> None:
    """Per-channel absmax weight quantization on every ``nn.Linear``."""
    fq = _load_vendored()
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            fq.quantize_weight_per_channel_absmax(module.weight, n_bits=wbits)


@torch.no_grad()
def apply_smoothquant_uniform(
    model,
    calib_batches: List[torch.Tensor],
    wbits: int = 8,
    alpha: float = 0.5,
    verbose: bool = True,
) -> dict:
    """SmoothQuant uniform baseline.

    1. Compute per-channel activation scales from the calibration batches.
    2. Smooth weights (redistribute quantization difficulty from activations
       to weights via scaling).
    3. Uniform per-channel absmax weight quantization at ``wbits``.

    Parameters
    ----------
    model : nn.Module
        HuggingFace causal-LM loaded in fp16/bf16/fp32.
    calib_batches : list[Tensor]
        Pre-tokenised input-id batches from the benchmark's calibration set.
    wbits : int
        Target weight bit-width (SmoothQuant paper uses 8; can go lower).
    alpha : float
        Smoothing strength.  0.5 = balanced; larger = smoother activations.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Empty – no per-layer measured statistics needed for uniform quantisation.
    """
    if verbose:
        n = len(calib_batches)
        print(f"[SmoothQuant] Computing activation scales from {n} calibration "
              f"batches ...")
    act_scales = _compute_act_scales(model, calib_batches)

    if verbose:
        print(f"[SmoothQuant] Smoothing weights (alpha={alpha}) ...")
    _smooth_weights(model, act_scales, alpha)

    if verbose:
        print(f"[SmoothQuant] Quantizing all Linear layers to "
              f"{wbits}-bit per-channel absmax ...")
    _quantize_all_linear(model, wbits)

    if verbose:
        print("[SmoothQuant] Done.")

    return {}
