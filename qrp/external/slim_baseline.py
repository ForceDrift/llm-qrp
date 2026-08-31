import importlib.util
import os
import random
import sys

import torch
import torch.nn as nn

from qrp.model_mapper import get_model_layers


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SLIM_DIR = os.path.join(_REPO_ROOT, "external", "slim-llm")

_vendored = {}


def _load_vendored():
    """Load external/slim-llm's core (slim_gptq.py + utils/) without importing
    their entry script or any of the vendored AutoGPTQ / lm-eval trees."""
    global _vendored
    if _vendored:
        return _vendored["slim_gptq"]

    if not os.path.isdir(SLIM_DIR):
        raise FileNotFoundError(
            f"Vendored SliM-LLM not found at {SLIM_DIR}.\n"
            "Clone it with: git clone https://github.com/Aaronhuang-778/SliM-LLM external/slim-llm"
        )

    def make_pkg(name, rel_path):
        pkg = types_module(name)
        pkg.__path__ = [os.path.join(SLIM_DIR, rel_path)]
        sys.modules[name] = pkg
        return pkg

    def types_module(name):
        import types
        return types.ModuleType(name)

    def _load(modname, rel_path):
        path = os.path.join(SLIM_DIR, rel_path)
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        return spec, mod

    make_pkg("qrp_external_slim", "")
    make_pkg("qrp_external_slim.utils", "utils")

    # The vendored modules import their siblings by bare name
    # (`from utils.X import ...`) — including the utils modules themselves —
    # so create every module object first, install bare-name aliases, and only
    # then execute in dependency order.
    loaded = {}
    for name in ("reconstruct", "salient_mask", "bitsearch", "mixed_quantizer"):
        spec, mod = _load(f"qrp_external_slim.utils.{name}", f"utils/{name}.py")
        setattr(sys.modules["qrp_external_slim.utils"], name, mod)
        sys.modules[f"utils.{name}"] = mod
        loaded[name] = (spec, mod)
    had_utils = "utils" in sys.modules
    saved_utils = sys.modules.get("utils")
    if not had_utils:
        sys.modules["utils"] = sys.modules["qrp_external_slim.utils"]
    if saved_utils is None:
        # keep the alias installed for later lazy imports
        sys.modules.setdefault("utils", sys.modules["qrp_external_slim.utils"])
    for name in ("reconstruct", "salient_mask", "bitsearch", "mixed_quantizer"):
        spec, mod = loaded[name]
        spec.loader.exec_module(mod)
    spec, mod = _load("qrp_external_slim.slim_gptq", "slim_gptq.py")
    spec.loader.exec_module(mod)

    _vendored.update(slim_gptq=mod,
                     utils={k: v[1] for k, v in loaded.items()})
    mod.utils = _vendored["utils"]
    return mod


def _ensure_cuda_noop():
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.empty_cache = lambda *a, **k: None


@torch.no_grad()
def apply_slim_uniform(
    model,
    calib_batches,
    wbits=2,
    groupsize=128,
    percdamp=0.01,
    metric="mse",
    lambda_salience=1.0,
    verbose=True,
):
    """Apply uniform SliM-LLM to every linear inside each decoder block, in place.

    Mirrors external/slim-llm's run.quant_sequential with the default
    activation-aware block allocation (salient_block=-1): each weight group of
    `groupsize` input columns gets wbits-1 / wbits / wbits+1 bits based on
    salience ranking and KL-divergence search, so the average bit-width stays
    ~wbits. Uses qrp's architecture-generic layer access instead of their
    Llama/OPT-specific utilities.
    """
    slim_gptq_mod = _load_vendored()
    _ensure_cuda_noop()

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
        raise RuntimeError("SliM-LLM calibration failed: no activations were captured.")

    sublayer_counter = 0
    for i in range(num_layers):
        block = layers[i]
        linears = {
            name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)
        }
        if not linears:
            continue

        slims = {}
        for name, lin in linears.items():
            sublayer_counter += 1
            quantizer = slim_gptq_mod.utils["mixed_quantizer"].Quantizer(
                lin.weight,
                method=f"{wbits}bit",
                groupsize=groupsize,
                metric=metric,
                lambda_salience=lambda_salience,
            )
            slims[name] = slim_gptq_mod.SliMGPTQ(
                lin,
                quantizer,
                disable_gptq=False,
                layer_index=sublayer_counter,
                salient_block=-1,
                nonsalient_block=-1,
                bit_width=wbits,
            )

        # Pass 1: accumulate Hessians.
        handles = []
        def mk_add(h):
            def tmp(_, inp, out):
                h.add_batch(inp[0].data, out.data)
            return tmp
        for h in slims.values():
            handles.append(h.layer.register_forward_hook(mk_add(h)))
        for hidden, kwargs in captured:
            block(hidden, **kwargs)
        for h in handles:
            h.remove()

        # Pass 2: activation-aware search for how many groups go up/down a bit.
        for h in slims.values():
            h.get_salience(blocksize=groupsize)
        handles = []
        def mk_block(h):
            def tmp(_, inp, out):
                h.get_block(inp[0].data, out.data, blocksize=groupsize)
            return tmp
        for h in slims.values():
            handles.append(h.layer.register_forward_hook(mk_block(h)))
        for hidden, kwargs in captured:
            block(hidden, **kwargs)
        for h in handles:
            h.remove()

        for name, h in slims.items():
            if verbose:
                print(f"[SliM-LLM] layer {i:>2}  quantizing {name}")
            h.fasterquant(
                blocksize=groupsize,
                percdamp=percdamp,
                saved_block_precision=None,
            )
            h.free()
        del slims

        next_captured = []
        for hidden, kwargs in captured:
            out = block(hidden, **kwargs)
            hs = out[0] if isinstance(out, tuple) else out
            next_captured.append((hs.detach(), kwargs))
        captured = next_captured

    model.config.use_cache = use_cache
