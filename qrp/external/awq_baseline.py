import gc
import importlib.util
import os
import sys
import types
from collections import defaultdict

import torch
import torch.nn as nn

from qrp.model_mapper import get_layer_structure, get_model_layers


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AWQ_DIR = os.path.join(_REPO_ROOT, "external", "awq")

_vendored = {}


def _load_vendored():
    """Load external/awq's device-agnostic core (pseudo_quantize_tensor)
    without triggering the CUDA-kernel-heavy awq package __init__."""
    global _vendored
    if _vendored:
        return _vendored["pseudo_quantize"]

    if not os.path.isdir(AWQ_DIR):
        raise FileNotFoundError(
            f"Vendored AWQ not found at {AWQ_DIR}.\n"
            "Clone it with: git clone https://github.com/mit-han-lab/llm-awq.git external/awq"
        )

    def make_pkg(name, rel):
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(AWQ_DIR, rel)]
        sys.modules.setdefault(name, mod)
        return mod

    def load_module(full_name, rel_path):
        path = os.path.join(AWQ_DIR, rel_path)
        spec = importlib.util.spec_from_file_location(full_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        return mod

    make_pkg("qrp_external_awq", "awq")
    make_pkg("qrp_external_awq.utils", "awq/utils")
    make_pkg("qrp_external_awq.quantize", "awq/quantize")

    # qmodule.py imports the AWQ CUDA inference engine at module level; we only
    # use its pure-torch ScaledActivation, so stub the engine if unavailable.
    try:
        import awq_inference_engine  # noqa: F401
    except ImportError:
        sys.modules["awq_inference_engine"] = types.ModuleType("awq_inference_engine")

    load_module("qrp_external_awq.utils.module", "awq/utils/module.py")
    load_module("qrp_external_awq.quantize.qmodule", "awq/quantize/qmodule.py")
    quantizer_mod = load_module(
        "qrp_external_awq.quantize.quantizer", "awq/quantize/quantizer.py")

    _vendored["pseudo_quantize"] = quantizer_mod.pseudo_quantize_tensor
    return _vendored["pseudo_quantize"]


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


def _is_norm(module):
    if isinstance(module, nn.LayerNorm):
        return True
    w = getattr(module, "weight", None)
    return (
        w is not None
        and w.dim() == 1
        and not hasattr(module, "in_features")
        and callable(getattr(module, "forward", None))
    )


def _to_ref(t, ref):
    return t.to(ref.device).to(ref.dtype)


def _name_of(named_linears, module):
    for name, mod in named_linears.items():
        if mod is module:
            return name
    return None


@torch.no_grad()
def _search_module_scale(inspect_module, linears2scale, x, kwargs,
                         pseudo_quantize, n_bit, q_group_size, n_grid=20):
    """Port of AWQ's _search_module_scale (external/awq auto_scale.py),
    device-agnostic."""
    import inspect as _inspect
    x = x.to(next(inspect_module.parameters()).device)
    fwd_sig = _inspect.signature(inspect_module.forward)
    safe_kwargs = {k: v for k, v in kwargs.items() if k in fwd_sig.parameters}
    org_out = inspect_module(x, **safe_kwargs)
    if isinstance(org_out, tuple):
        org_out = org_out[0]

    x_max = _get_act_scale(x)

    best_error = float("inf")
    best_ratio = -1
    best_scales = None

    org_sd = {k: v.cpu() for k, v in inspect_module.state_dict().items()}
    for i in range(n_grid):
        ratio = i * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for fc in linears2scale:
            fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
            fc.weight.data = pseudo_quantize(
                fc.weight.data, n_bit=n_bit, zero_point=True,
                q_group_size=q_group_size,
            ) / scales.view(1, -1).to(fc.weight.device)
        out = inspect_module(x, **safe_kwargs)
        if isinstance(out, tuple):
            out = out[0]

        loss = (org_out - out).float().pow(2).mean().item()
        if loss < best_error:
            best_error = loss
            best_ratio = ratio
            best_scales = scales
        inspect_module.load_state_dict(org_sd)

    if best_ratio == -1:
        raise RuntimeError("AWQ scale search found no valid ratio.")

    return best_scales.view(-1).detach()


@torch.no_grad()
def _apply_scale(prev_op, layers, scales):
    """Port of AWQ's scale_ln_fcs / scale_fc_fc, device-agnostic."""
    if isinstance(prev_op, nn.Linear):
        assert len(layers) == 1
        fc1, fc2 = prev_op, layers[0]
        scales = _to_ref(scales, fc1.weight)
        fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
        if fc1.bias is not None:
            fc1.bias.div_(scales.view(-1))
        fc2.weight.mul_(scales.view(1, -1))
    elif _is_norm(prev_op):
        ln, fcs = prev_op, layers
        scales = _to_ref(scales, ln.weight)
        ln.weight.div_(scales)
        if hasattr(ln, "bias") and ln.bias is not None:
            ln.bias.div_(scales)
        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))
    else:
        raise NotImplementedError(f"Unsupported prev op type {type(prev_op)}")


@torch.no_grad()
def _auto_clip_layer(w, input_feat, pseudo_quantize, n_bit, q_config,
                     n_grid=20, max_shrink=0.5, n_sample_token=512):
    """Port of AWQ's auto_clip_layer (external/awq auto_clip.py),
    device-agnostic."""
    assert w.dim() == 2
    orig_in_features = w.shape[1]
    group_size = q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    _pad = 0
    if w.shape[1] % group_size != 0:
        _pad = group_size - (w.shape[1] % group_size)
        w = torch.nn.functional.pad(w, (0, _pad))
        input_feat = torch.nn.functional.pad(input_feat, (0, _pad))
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    step = max(1, input_feat.shape[1] // n_sample_token)
    input_feat = input_feat[:, ::step]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64
    if w.shape[0] % oc_batch_size != 0:
        oc_batch_size = max(1, min(oc_batch_size, w.shape[0]))

    w_all = w
    best_max_val_all = []

    for i_b in range((w_all.shape[0] + oc_batch_size - 1) // oc_batch_size):
        w = w_all[i_b * oc_batch_size:(i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize(cur_w, n_bit=n_bit, **q_config)
            cur_out = (input_feat * q_w).sum(dim=-1)

            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)
    _empty_cache()
    best_max_val = best_max_val.squeeze(1).squeeze(-1)
    best_max_val = best_max_val.unsqueeze(-1).expand(-1, -1, group_size)
    best_max_val = best_max_val.reshape(best_max_val.shape[0], -1)
    if _pad:
        best_max_val = best_max_val[:, :orig_in_features]
    return best_max_val


def _discover_smoothing_groups(block, named_linears):
    """Find (prev_op, [target linears], inspect_module, feat_key) quadruples,
    generic across architectures known to qrp.model_mapper."""
    (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(block)

    def resolve(parent, proj_name):
        return getattr(parent, proj_name, None) if parent is not None else None

    def find_norm(names):
        for n in names:
            cand = getattr(block, n, None)
            if cand is not None and _is_norm(cand):
                return cand
        return None

    groups = []

    if attn_parent is not None:
        norm_attn = find_norm([
            "input_layernorm", "attention_layernorm", "norm1", "ln_1", "ln_attn",
        ])
        inputs = [resolve(attn_parent, p) for p in attn_projs[:-1]]
        inputs = [m for m in inputs if m is not None]
        o_proj = resolve(attn_parent, attn_projs[-1]) if len(attn_projs) >= 2 else None
        if norm_attn is not None and inputs and o_proj is not None:
            groups.append((norm_attn, inputs, attn_parent,
                           _name_of(named_linears, inputs[0])))

        if len(attn_projs) >= 4:
            v_proj = resolve(attn_parent, attn_projs[-2])
            if v_proj is not None and o_proj is not None \
                    and v_proj.weight.shape == o_proj.weight.shape:
                groups.append((v_proj, [o_proj], o_proj,
                               _name_of(named_linears, o_proj)))

    if mlp_parent is not None:
        norm_mlp = find_norm([
            "post_attention_layernorm", "ffn_layernorm", "norm2", "ln_2", "ln_mlp",
        ])
        downs = ["down_proj", "w2", "output_linear"]
        ups = [resolve(mlp_parent, p) for p in mlp_projs if p not in downs]
        ups = [m for m in ups if m is not None]
        down = None
        for d in downs:
            down = resolve(mlp_parent, d)
            if down is not None:
                break
        if norm_mlp is not None and ups and down is not None:
            groups.append((norm_mlp, ups, mlp_parent,
                           _name_of(named_linears, ups[0])))

        if ups and down is not None:
            up_main = ups[-1]
            if up_main.weight.shape[-1] == down.weight.shape[0]:
                groups.append((up_main, [down], down,
                               _name_of(named_linears, down)))

    return [g for g in groups if g[3] is not None]


@torch.no_grad()
def apply_awq_uniform(
    model,
    calib_batches,
    wbits=4,
    q_group_size=128,
    mse_range=True,
    verbose=True,
):
    """Apply activation-aware weighted quantization (AWQ) uniformly over all
    decoder blocks, in place. Mirrors external/awq's run_awq +
    pseudo_quantize_model_weight flow, but device-agnostic."""
    pseudo_quantize = _load_vendored()

    q_config = {"zero_point": True, "q_group_size": q_group_size}

    layers = get_model_layers(model)
    num_layers = len(layers)

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    captured = []

    def pre_hook(module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("use_cache", None)
        captured.append((args[0].detach(), kwargs))

    handle = layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    for batch in calib_batches:
        model(batch)
    handle.remove()

    if not captured:
        raise RuntimeError("AWQ calibration failed: no activations were captured.")

    for i in range(num_layers):
        block = layers[i]
        named_linears = {
            name: m for name, m in block.named_modules() if isinstance(m, nn.Linear)
        }
        if not named_linears:
            continue

        input_feat = defaultdict(list)
        handles = []
        for name, lin in named_linears.items():
            def mk(n):
                def tmp(_, inp, out):
                    input_feat[n].append(
                        inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())
                return tmp
            handles.append(lin.register_forward_hook(mk(name)))

        next_captured = []
        for hidden, kwargs in captured:
            out = block(hidden, **kwargs)
            hs = out[0] if isinstance(out, tuple) else out
            next_captured.append((hs.detach(), kwargs))

        for h in handles:
            h.remove()
        feats = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        del input_feat
        _empty_cache()

        smoothed_any = False
        groups = _discover_smoothing_groups(block, named_linears)
        for gi, (prev_op, targets, inspect_module, feat_key) in enumerate(groups):
            try:
                scales = _search_module_scale(
                    inspect_module, targets, feats[feat_key],
                    captured[0][1], pseudo_quantize, wbits, q_group_size,
                )
            except (NotImplementedError, RuntimeError, KeyError) as exc:
                if verbose:
                    print(f"[AWQ] layer {i:>2} group {gi}: skipped ({exc})")
                continue
            _apply_scale(prev_op, targets, scales)
            for t in targets:
                tn = _name_of(named_linears, t)
                if tn is not None and tn in feats:
                    feats[tn].div_(_to_ref(scales.view(1, -1), feats[tn]))
            smoothed_any = True
            del scales
            _empty_cache()

        if mse_range and smoothed_any:
            clip_list = []
            for name, lin in named_linears.items():
                if any(s in name for s in ["q_", "k_", "query", "key", "Wqkv"]):
                    continue
                if name not in feats:
                    continue
                max_val = _auto_clip_layer(
                    lin.weight.data, feats[name], pseudo_quantize, wbits, q_config
                )
                clip_list.append((lin, max_val))
            for lin, max_val in clip_list:
                max_val = max_val.to(lin.weight.device).to(lin.weight.dtype)
                lin.weight.data = torch.clamp(lin.weight.data, -max_val, max_val)
            del clip_list
            _empty_cache()

        for name, lin in named_linears.items():
            try:
                lin.weight.data = pseudo_quantize(lin.weight.data, n_bit=wbits, **q_config)
            except AssertionError:
                if verbose:
                    print(f"[AWQ] layer {i:>2} {name}: group size mismatch, "
                          f"falling back to per-tensor")
                lin.weight.data = pseudo_quantize(
                    lin.weight.data, n_bit=wbits, zero_point=True, q_group_size=-1)

        if verbose:
            print(f"[AWQ] layer {i:>2} done "
                  f"(smoothing={'on' if smoothed_any else 'off'})")
        captured = next_captured

    model.config.use_cache = use_cache
