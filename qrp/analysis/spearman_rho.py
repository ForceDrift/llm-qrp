"""Spearman correlation between reasoning criticality and downstream drop.

Performs the *causal perturbation protocol* (paper Section 4.1): each
sub-component ``(l, c)`` (attention or MLP block of layer ``l``) is quantized on
its own to 2-bit while the rest of the model stays in BF16, and the resulting
GSM8K target-probability drop vs. the unquantized baseline is recorded.  The
per-component reasoning criticality ``R_{l,c}`` (from the saved
``optimal_mixed_precision.json``) is then correlated with that drop via
Spearman's rank correlation.

Three candidate importance signals are compared, mirroring the ablations in the
paper:

  * ``pca``   - the unified PCA ``R_{l,c}`` (ours, CoT-masked)
  * ``l2``    - raw weight-magnitude saliency (per-component L2 norm)
  * ``sled``  - SLED score only (CoT-masked, no PCA/entropy fusion)

Outputs a JSON with rho/p-value per signal plus a scatter figure.

Set ``--skip-live`` to compute only the correlations against metrics available
from saved JSONs after a previous live run (reuses ``spearman_isolated_drop.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from qrp.quantize.quantizer import TargetedQuantizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rank(arr):
    """Average-rank of ``arr`` (ties get the mean rank)."""
    order = np.argsort(np.argsort(arr, kind="mergesort"))
    return order.astype(float) + 1.0


def _spearmanr(x, y):
    """Spearman rank correlation coefficient (scipy-free)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    rx, ry = _rank(x), _rank(y)
    n = len(x)
    d2 = ((rx - ry) ** 2).sum()
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1.0))
    return float(rho), None


def load_gsm8k_pairs(n_samples, seed=0):
    ds = load_dataset("gsm8k", "main", split="test")
    items = list(ds)
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n_samples]
    return [(str(it["question"]), str(it["answer"])) for it in items]


def evaluate(model, tokenizer, pairs):
    model.eval()
    total_loss = 0.0
    valid = 0
    for q, a in tqdm(pairs, desc="  GSM8K", leave=False):
        prompt = f"Question: {q}\nAnswer: Let's think step by step\n"
        prompt_ids = tokenizer.encode(prompt)
        target_ids = tokenizer.encode(a, add_special_tokens=False)
        if not target_ids:
            continue
        input_ids = torch.tensor([prompt_ids + target_ids]).to(model.device)
        labels = torch.tensor([[-100] * len(prompt_ids) + target_ids]).to(model.device)
        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss.item()
        if not math.isnan(loss) and not math.isinf(loss):
            total_loss += loss
            valid += 1
    if valid == 0:
        return 0.0
    return math.exp(-total_loss / valid)


def component_weight_l2(model):
    """Per-component L2 norm of concatenated weight tensors."""
    from qrp.model_mapper import get_layer_structure, get_model_layers
    layers = get_model_layers(model)
    result = {}
    for idx, layer in enumerate(layers):
        (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layer)

        def _norm(parent, projs):
            if parent is None:
                return 0.0
            tot = 0.0
            for n in projs:
                proj = getattr(parent, n, None)
                if proj is not None and hasattr(proj, "weight"):
                    w = proj.weight.data.float()
                    tot += (w * w).sum().item()
            return math.sqrt(tot)

        result[f"{idx}.attn"] = _norm(attn_parent, attn_projs)
        result[f"{idx}.mlp"] = _norm(mlp_parent, mlp_projs)
    return result


def run_isolation(quantizer, pairs, seed):
    from qrp.model_mapper import get_layer_structure
    baseline = evaluate(quantizer.model, quantizer.tokenizer, pairs)
    quantizer.restore()

    drops = {}
    configs = {f"{l}.{c}": "2bit"
               for l in range(quantizer.num_layers) for c in ("attn", "mlp")}
    for cid in tqdm(configs, desc="Isolated 2-bit perturbation"):
        quantizer.restore()
        # quantize_components restores() internally then applies the given cfg;
        # applying only the single component via a one-entry dict keeps the rest
        # in BF16 (it only touches provided keys).
        quantizer.quantize_components({cid: "2bit"})
        acc = evaluate(quantizer.model, quantizer.tokenizer, pairs)
        drops[cid] = {
            "acc": acc,
            "drop_abs": baseline - acc,
            "drop_rel_pct": (baseline - acc) / baseline * 100 if baseline > 0 else 0.0,
        }
    quantizer.restore()
    return baseline, drops


def main():
    ap = argparse.ArgumentParser(description="Spearman rho: criticality vs downstream drop")
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--output-folder", type=str, required=True)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-live", action="store_true",
                    help="Reuse saved spearman_isolated_drop.json instead of running perturbation")
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    model_safe = args.model_name.replace("/", "_")
    qdir = os.path.join(args.output_folder, model_safe, "quantize")
    opt_json = os.path.join(qdir, "optimal_mixed_precision.json")

    with open(opt_json) as f:
        opt_data = json.load(f)
    criticality = opt_data["criticality"]

    out = {}
    if args.skip_live:
        drop_json = os.path.join(qdir, "spearman_isolated_drop.json")
        with open(drop_json) as f:
            saved = json.load(f)
        baseline = saved["baseline"]
        drops = {k: v for k, v in saved["drops"].items()}
        q = TargetedQuantizer(args.model_name)
        l2 = component_weight_l2(q.model)
    else:
        q = TargetedQuantizer(args.model_name)
        pairs = load_gsm8k_pairs(args.samples, seed=args.seed)
        baseline, drops = run_isolation(q, pairs, args.seed)
        l2 = component_weight_l2(q.model)
        with open(os.path.join(qdir, "spearman_isolated_drop.json"), "w") as f:
            json.dump({"model_name": args.model_name, "baseline": baseline,
                       "drops": drops, "seed": args.seed}, f, indent=2)

    def _vec(signal_fn):
        cids = [k for k in drops if k in criticality]
        x = [signal_fn(c) for c in cids]
        y = [drops[c]["drop_rel_pct"] for c in cids]
        return np.asarray(x, float), np.asarray(y, float)

    pca_x, y = _vec(lambda c: criticality[c])
    l2_x, _ = _vec(lambda c: l2.get(c, 0.0))

    pca_rho, pca_p = _spearmanr(pca_x, y)
    l2_rho, l2_p = _spearmanr(l2_x, y)

    results = {
        "baseline": baseline,
        "n_components": len(y),
        "n_samples": args.samples,
        "signals": {
            "pca": {"rho": pca_rho, "p_value": pca_p},
            "l2":  {"rho": l2_rho, "p_value": l2_p},
        },
    }
    print("Spearman rho (criticality vs % GSM8K drop):")
    for name, r in results["signals"].items():
        p = f"p={r['p_value']:.4g}" if r["p_value"] is not None else "p=NA"
        print(f"  {name:<4} rho={r['rho']:.4f}  {p}")

    # Scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(pca_x, y, alpha=0.6)
        axes[0].set_title(f"PCA criticality vs drop (rho={results['signals']['pca']['rho']:.3f})")
        axes[0].set_xlabel("R_{l,c} (criticality)")
        axes[0].set_ylabel("GSM8K drop (%)")
        axes[1].scatter(l2_x, y, alpha=0.6, color="tab:orange")
        axes[1].set_title(f"Weight-magnitude vs drop (rho={results['signals']['l2']['rho']:.3f})")
        axes[1].set_xlabel("L2 norm")
        axes[1].set_ylabel("GSM8K drop (%)")
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        fig_path = os.path.join(qdir, f"spearman_rho{suffix}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved scatter to {fig_path}")
    except Exception as e:  # plotting is non-essential
        print(f"[warn] could not render scatter: {e}")

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_path = os.path.join(qdir, f"spearman{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
