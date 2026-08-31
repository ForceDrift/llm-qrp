import importlib.util
import os
import sys

import torch
import torch.nn as nn

from qrp.model_mapper import get_model_layers


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPQR_DIR = os.path.join(_REPO_ROOT, "external", "spqr")

_vendored = {}


def _load_vendored():
    """Load external/spqr's quantization core (quant_groups.py,
    weight_permutation.py, spqr_engine.py) without importing the repo's
    top-level scripts or its vendored lm-evaluation-harness."""
    global _vendored
    if _vendored:
        return _vendored["engine"]

    if not os.path.isdir(SPQR_DIR):
        raise FileNotFoundError(
            f"Vendored SpQR not found at {SPQR_DIR}.\n"
            "Clone it with: git clone https://github.com/Vahe1994/SpQR external/spqr"
        )

    def _load(name):
        path = os.path.join(SPQR_DIR, f"{name}.py")
        modname = f"qrp_external_spqr.{name}"
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        return spec, mod

    spec_qg, mod_qg = _load("quant_groups")
    spec_qg.loader.exec_module(mod_qg)

    # weight_permutation.py applies @torch.jit.script at import time; skip the
    # JIT compile here (it is only exercised by the "spearman" permutation).
    prev_script = getattr(torch.jit, "script", None)
    torch.jit.script = lambda fn: fn
    try:
        spec_wp, mod_wp = _load("weight_permutation")
        spec_wp.loader.exec_module(mod_wp)
    finally:
        if prev_script is not None:
            torch.jit.script = prev_script

    # spqr_engine.py imports its siblings by bare module name.
    aliases = {"quant_groups": mod_qg, "weight_permutation": mod_wp}
    saved = {k: sys.modules.get(k) for k in aliases}
    sys.modules.update(aliases)
    try:
        spec_e, mod_e = _load("spqr_engine")
        spec_e.loader.exec_module(mod_e)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    _vendored.update(engine=mod_e)
    return mod_e


@torch.no_grad()
def apply_spqr_uniform(
    model,
    calib_batches,
    wbits=3,
    groupsize=16,
    qq_scale_bits=3,
    qq_zero_bits=3,
    qq_groupsize=16,
    outlier_threshold=0.2,
    permutation_order="act_order",
    percdamp=1.0,
    verbose=True,
):
    """Apply uniform SpQR to every linear inside each decoder block, in place.

    Mirrors external/spqr's main.quantize_spqr (true_sequential disabled:
    Hessians for all sublayers are accumulated from full-block forwards), but
    uses qrp's architecture-generic layer access instead of their
    Llama/OPT/Falcon-specific utilities.

    Returns {"outlier_share": float, "effective_bpw": float} aggregated over
    every quantized sublayer.
    """
    engine = _load_vendored()

    layers = get_model_layers(model)
    num_layers = len(layers)

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    captured = []

    def pre_hook(module, args, kwargs):
        captured.append((args[0].detach(), dict(kwargs)))

    handle = layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    for batch in calib_batches:
        model(batch)
    handle.remove()

    if not captured:
        raise RuntimeError("SpQR calibration failed: no activations were captured.")

    total_outliers = 0
    total_weights = 0

    for i in range(num_layers):
        block = layers[i]
        linears = {
            name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)
        }
        if not linears:
            continue

        handlers = {
            name: engine.SPQRUtil(lin) for name, lin in linears.items()
        }

        handles = []
        def mk(h):
            def tmp(_, inp, out):
                h.add_batch(inp[0].data)
            return tmp
        for h in handlers.values():
            handles.append(h.layer.register_forward_hook(mk(h)))

        for hidden, kwargs in captured:
            block(hidden, **kwargs)

        for h in handles:
            h.remove()

        for name, h in handlers.items():
            if verbose:
                print(f"[SpQR] layer {i:>2}  quantizing {name}")
            result = h.quantize(
                bits=wbits,
                blocksize=128,
                percdamp=percdamp,
                groupsize=groupsize,
                sym=False,
                perchannel=True,
                round_zero=False,
                qq_scale_bits=qq_scale_bits,
                qq_zero_bits=qq_zero_bits,
                qq_groupsize=qq_groupsize,
                outlier_relative_threshold=outlier_threshold,
                permutation_order=permutation_order,
                simplified_outliers=False,
                verbose=False,
            )
            h.layer.weight.data = result.weight.to(
                h.layer.weight.data.dtype
            )
            n_ol = int(result.unstructured_outlier_mask.sum())
            total_outliers += n_ol
            total_weights += result.weight.numel()
            if verbose:
                print(f"[SpQR] layer {i:>2}  {name}: outliers "
                      f"{n_ol / max(result.weight.numel(), 1):.2%}")
        del handlers

        next_captured = []
        for hidden, kwargs in captured:
            out = block(hidden, **kwargs)
            hs = out[0] if isinstance(out, tuple) else out
            next_captured.append((hs.detach(), kwargs))
        captured = next_captured

    model.config.use_cache = use_cache

    outlier_share = total_outliers / max(total_weights, 1)
    effective_bpw = average_bits_per_weight(
        wbits=wbits,
        groupsize=groupsize,
        qq_scale_bits=qq_scale_bits,
        qq_zero_bits=qq_zero_bits,
        qq_groupsize=qq_groupsize,
        round_zero=False,
        outlier_share=outlier_share,
    )
    return {"outlier_share": outlier_share, "effective_bpw": effective_bpw}


def average_bits_per_weight(
    wbits=3,
    groupsize=16,
    qq_scale_bits=3,
    qq_zero_bits=3,
    qq_groupsize=16,
    round_zero=False,
    outlier_share=0.0,
):
    """Port of external/spqr main.get_average_number_of_bits."""
    qs = qq_scale_bits if qq_scale_bits else 16
    qz = qq_zero_bits if qq_zero_bits else 16
    gs = groupsize if groupsize else float("inf")
    qq_gs = qq_groupsize if qq_groupsize else float("inf")

    if round_zero:
        bpw = wbits + (qs + wbits) / gs + (16 + 16) / (gs * qq_gs)
    else:
        bpw = wbits + (qs + qz) / gs + 2 * (16 + 16) / (gs * qq_gs)

    return bpw + 32 * outlier_share


def estimate_spqr_size_bytes(
    model,
    num_layers,
    *,
    wbits=3,
    groupsize=16,
    qq_scale_bits=3,
    qq_zero_bits=3,
    qq_groupsize=16,
    round_zero=False,
    outlier_share=0.0,
):
    """Estimated stored size of the uniformly SpQR-quantized decoder blocks.

    Counts the same parameters as the uniform-bit baselines and prices them
    with the average-bits-per-weight formula from external/spqr (base weights,
    double-quantized scale/zero statistics, fp16 sparse outliers)."""
    from qrp.model_mapper import get_layer_structure

    bpw = average_bits_per_weight(
        wbits=wbits,
        groupsize=groupsize,
        qq_scale_bits=qq_scale_bits,
        qq_zero_bits=qq_zero_bits,
        qq_groupsize=qq_groupsize,
        round_zero=round_zero,
        outlier_share=outlier_share,
    )

    total = 0.0
    layers = get_model_layers(model)
    for idx in range(num_layers):
        (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layers[idx])
        for parent, proj_names in ((attn_parent, attn_projs), (mlp_parent, mlp_projs)):
            if parent is None:
                continue
            for n in proj_names:
                proj = getattr(parent, n, None)
                if proj is not None and hasattr(proj, "weight"):
                    total += proj.weight.numel()
    return total * bpw / 8.0
