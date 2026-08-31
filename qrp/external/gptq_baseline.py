import importlib.util
import os
import random
import sys

import torch
import torch.nn as nn
from datasets import load_dataset

from qrp.model_mapper import get_model_layers


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GPTQ_DIR = os.path.join(_REPO_ROOT, "external", "gptq")

_vendored = {}


def _load_vendored():
    global _vendored
    if _vendored:
        return _vendored["quant"], _vendored["gptq"]

    if not os.path.isdir(GPTQ_DIR):
        raise FileNotFoundError(
            f"Vendored GPTQ not found at {GPTQ_DIR}.\n"
            "Clone it with: git clone https://github.com/ist-daslab/gptq external/gptq"
        )

    def _load(name):
        path = os.path.join(GPTQ_DIR, f"{name}.py")
        modname = f"qrp_external_gptq.{name}"
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        return spec, mod

    spec_q, mod_q = _load("quant")
    spec_q.loader.exec_module(mod_q)

    prev_quant = sys.modules.get("quant")
    sys.modules["quant"] = mod_q
    try:
        spec_g, mod_g = _load("gptq")
        spec_g.loader.exec_module(mod_g)
    finally:
        if prev_quant is None:
            sys.modules.pop("quant", None)
        else:
            sys.modules["quant"] = prev_quant

    _vendored.update(quant=mod_q, gptq=mod_g)
    return mod_q, mod_g


def _ensure_cuda_sync_noop():
    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **k: None


def build_calibration_batches(tokenizer, nsamples=32, seqlen=2048, seed=0, device="cpu"):
    ds = load_dataset("gsm8k", "main", split="train")
    rng = random.Random(seed)
    texts = [
        f"Question: {r['question']}\nAnswer: Let's think step by step\n{r['answer']}"
        for r in ds
    ]
    rng.shuffle(texts)

    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
        if len(ids) >= nsamples * seqlen:
            break

    batches = []
    for i in range(nsamples):
        chunk = ids[i * seqlen:(i + 1) * seqlen]
        if len(chunk) < seqlen:
            break
        batches.append(torch.tensor([chunk], device=device))
    return batches


@torch.no_grad()
def apply_gptq_uniform(
    model,
    calib_batches,
    wbits=4,
    symmetric=True,
    percdamp=0.01,
    groupsize=-1,
    actorder=False,
    verbose=True,
):
    """Apply uniform GPTQ to every linear inside each decoder block, in place."""
    quant_mod, gptq_mod = _load_vendored()
    _ensure_cuda_sync_noop()

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
        raise RuntimeError("GPTQ calibration failed: no activations were captured.")

    for i in range(num_layers):
        block = layers[i]
        linears = {
            name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)
        }
        if not linears:
            continue

        gpts = {}
        for name, lin in linears.items():
            g = gptq_mod.GPTQ(lin)
            q = quant_mod.Quantizer()
            q.configure(wbits, perchannel=True, sym=symmetric, mse=False)
            g.quantizer = q
            gpts[name] = g

        handles = []
        for name, lin in linears.items():
            def mk(g):
                def tmp(_, inp, out):
                    g.add_batch(inp[0].data, out.data)
                return tmp
            handles.append(lin.register_forward_hook(mk(gpts[name])))

        for hidden, kwargs in captured:
            block(hidden, **kwargs)

        for h in handles:
            h.remove()

        for name, g in gpts.items():
            if verbose:
                print(f"[GPTQ] layer {i:>2}  quantizing {name}")
            g.fasterquant(
                blocksize=128, percdamp=percdamp,
                groupsize=groupsize, actorder=actorder,
            )
            g.free()
        del gpts

        next_captured = []
        for hidden, kwargs in captured:
            out = block(hidden, **kwargs)
            hs = out[0] if isinstance(out, tuple) else out
            next_captured.append((hs.detach(), kwargs))
        captured = next_captured

    model.config.use_cache = use_cache


def estimate_uniform_bits_size_bytes(model, num_layers, wbits):
    from qrp.model_mapper import get_layer_structure

    bytes_per_param = wbits / 8.0
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
    return total * bytes_per_param
