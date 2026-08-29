"""Sub-component mixed-precision allocation under a memory bit-budget.

Replaces the previous heuristic grid sweep over manual layer percentiles
``(pct_4bit, pct_8bit)`` with a mathematically grounded *constrained
optimization* at sub-component granularity.

Each decoder layer ``l`` is decomposed into two independently-quantizable
components ``c in {attn, mlp}``.  Every component carries a **criticality**
``R_{l,c}`` obtained from a *parameter-free* principal-component fusion of the
profiling signals (Section 3.3).  The normalized feature vector

    x_{l,c} = [tilde{S}_{SLED}(l,c), tilde{DeltaH}(l,c)]

(each signal z-scored across sub-components) is projected onto the first
principal component ``PC1`` learned from all sub-components jointly:

    R_{l,c} = w_1 . tilde{S}_{SLED}(l,c) + w_2 . tilde{DeltaH}(l,c),

with ``(w_1, w_2)`` the dominant eigenvector of the signal covariance matrix.
The weighting is learned directly from model dynamics -- no lambda-weights and
no min-max normalization anywhere in the framework.

The allocator then solves, for a target average bit budget ``B_target``
(bits/param):

    maximize    sum_{l,c,b} x_{l,c,b} . R_{l,c} . f_{l,c}(b)
    subject to  1/P_total * sum_{l,c,b} x_{l,c,b} . P_{l,c} . b <= B_target,
                sum_b x_{l,c,b} = 1,   x_{l,c,b} in {0, 1},

i.e. a 0-1 Multiple-Choice Knapsack over candidate bits ``{2, 3, 4, 8, 16}``.
The step-wise percentile grid search is removed.  Salient Outlier Channel
Protection extracts the top-0.1% highest-activation channels per sub-matrix
into an unquantized BF16 sparse matrix ``W_fp16`` (captured during profiling at
``outlier_channels.json``), so those channels never enter the low-bit weight
path.  Candidates form a Pareto frontier indexed by *memory density* and are
validated on GSM8K; the most efficient one above the accuracy floor is
selected.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from qrp.analysis.subcomponent_sled import COMPONENTS
from qrp.model_mapper import get_layer_structure, get_model_layers
from qrp.quantize.allocation_core import (  # noqa: F401  (re-exported API)
    BITS,
    BYTES_PER_PARAM,
    LOW3_MAX_FRAC,
    LOW6_MAX_FRAC,
    MpcAllocator,
    pca_criticality,
    sigmoid,
    zscore,
)
from qrp.quantize.quantizer import TargetedQuantizer


# --------------------------------------------------------------------------- #
# Per-component parameter estimation
# --------------------------------------------------------------------------- #
def component_params(model, layer_idx: int, component: str) -> int:
    layer = get_model_layers(model)[layer_idx]
    (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layer)
    parent, projs = (attn_parent, attn_projs) if component == "attn" else (mlp_parent, mlp_projs)
    total = 0
    if parent is not None:
        for name in projs:
            proj = getattr(parent, name, None)
            if proj is not None and hasattr(proj, "weight"):
                total += proj.weight.numel()
    return total


def estimate_component_params(model, num_layers: int) -> dict[str, int]:
    return {
        f"{l}.{c}": component_params(model, l, c)
        for l in range(num_layers)
        for c in COMPONENTS
    }


def estimate_actual_outlier_share(model, num_layers: int, outlier_channels: dict[str, list]) -> float:
    """Mean protected-channel fraction of the quantized input dims."""
    shares = []
    for cid, channels in outlier_channels.items():
        if not channels:
            continue
        layer_str, c = cid.rsplit(".", 1)
        layer = get_model_layers(model)[int(layer_str)]
        (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layer)
        parent, projs = (attn_parent, attn_projs) if c == "attn" else (mlp_parent, mlp_projs)
        in_features = []
        if parent is not None:
            for name in projs:
                proj = getattr(parent, name, None)
                if proj is not None and hasattr(proj, "in_features"):
                    in_features.append(proj.in_features)
        if in_features:
            shares.append(len(channels) / float(np.mean(in_features)))
    return float(np.mean(shares)) if shares else 0.001


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_gsm8k(model, tokenizer, dataset):
    total_loss = 0.0
    valid_samples = 0
    model.eval()

    for item in tqdm(dataset, desc="Evaluating GSM8K", leave=False):
        question = item["question"]
        expected_ans = item["answer"]
        prompt = f"Question: {question}\nAnswer: Let's think step by step\n"
        prompt_ids = tokenizer.encode(prompt)
        target_ids = tokenizer.encode(expected_ans, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + target_ids]).to(model.device)
        labels = torch.tensor([[-100] * len(prompt_ids) + target_ids]).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss.item()

        if not math.isnan(loss):
            total_loss += loss
            valid_samples += 1

    avg_loss = total_loss / valid_samples if valid_samples > 0 else float("inf")
    return math.exp(-avg_loss) if avg_loss != float("inf") else 0.0


def select_best(frontier_results, accuracy_floor, baseline_acc, baseline_size,
                lossless_tol=1.5):
    """Pick the most efficient candidate above the accuracy floor.

    If any quantized candidate is effectively near-lossless (accuracy within
    ``lossless_tol`` percent of baseline), prefer the highest-compression one of
    those, so noise-level differences near baseline don't pick a tiny-compression
    point that merely measured marginally above baseline.
    """
    for r in frontier_results:
        acc_drop = (1.0 - r["accuracy"] / baseline_acc) * 100.0 if baseline_acc > 0 else 0.0
        if acc_drop > 0.01:
            r["efficiency"] = r["size_reduction_pct"] / acc_drop
        elif r["size_reduction_pct"] > 0:
            r["efficiency"] = r["size_reduction_pct"] * 100.0
        else:
            r["efficiency"] = 0.0
        r["baseline_accuracy"] = baseline_acc
        r["accuracy_drop_pct"] = acc_drop

    valid = [r for r in frontier_results if r["accuracy"] >= accuracy_floor]
    if not valid:
        valid = frontier_results
    quantized = [r for r in valid if r["size_reduction_pct"] > 0.0]
    pool = quantized if quantized else valid

    within = [r for r in pool if r["accuracy_drop_pct"] <= lossless_tol]
    if within:
        return max(within, key=lambda x: x["size_reduction_pct"])
    return max(pool, key=lambda x: x["efficiency"])


def _plot_frontier(frontier_results, best, baseline_acc, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sizes = [r["size_reduction_pct"] for r in frontier_results]
    accs = [r["accuracy"] for r in frontier_results]
    ax.scatter(sizes, accs, c="steelblue", s=50, alpha=0.8, edgecolors="#333", linewidths=0.5)
    ax.scatter([best["size_reduction_pct"]], [best["accuracy"]],
               color="none", edgecolors="red", s=220, linewidths=2.5, zorder=5,
               label="Selected optimal")
    ax.axhline(baseline_acc, color="gray", linestyle=":", alpha=0.6, label="Baseline BF16")
    ax.set_xlabel("Size Reduction (%)")
    ax.set_ylabel("Target Probability (Accuracy)")
    ax.set_title("Sub-component Bit-budget Frontier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def default_profile_path(base_dir: str, fname: str) -> str:
    for root, _dirs, files in os.walk(base_dir):
        if fname in files:
            return os.path.join(root, fname)
    return os.path.join(base_dir, fname)


def load_profile_and_outliers(base_dir: str, profile_arg: str | None):
    profile_path = profile_arg or default_profile_path(base_dir, "subcomponent_scores.json")
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"Could not find {profile_path}. Run run_analysis.py with "
            "--granularity subcomponent --cot first."
        )
    with open(profile_path) as f:
        profile = json.load(f)

    oc_path = default_profile_path(base_dir, "outlier_channels.json")
    outlier_channels = {}
    if os.path.exists(oc_path):
        with open(oc_path) as f:
            oc = json.load(f)
        outlier_channels = oc.get("outlier_channels", {})
    return profile, outlier_channels, oc_path


def main():
    parser = argparse.ArgumentParser(
        description="Sub-component mixed-precision allocation under a bit budget"
    )
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to subcomponent_scores.json (defaults to <out>/<model>/*/subcomponent_scores.json)")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--bits-per-param", type=float, default=3.5,
                        help="Primary target average bit budget (bits/param)")
    parser.add_argument("--min-accuracy-floor", type=float, default=0.5)
    parser.add_argument("--bits-step", type=float, default=0.5,
                        help="Density step (bits/param) for the frontier sweep")
    parser.add_argument("--low3-max-frac", type=float, default=LOW3_MAX_FRAC,
                        help="Max fraction of components allowed at 3-bit (alloy cap)")
    parser.add_argument("--low6-max-frac", type=float, default=LOW6_MAX_FRAC,
                        help="Max fraction of components allowed at <=6-bit (alloy cap)")
    parser.add_argument("--target-densities", type=str, default=None,
                        help="Comma-separated explicit bit budgets (2.0-16.0)")
    parser.add_argument("--plot", action="store_true", help="Emit a frontier plot")
    args = parser.parse_args()

    model_name_safe = args.model_name.replace("/", "_")
    base_dir = os.path.join(args.output_folder, model_name_safe)

    profile, outlier_channels, oc_path = load_profile_and_outliers(base_dir, args.profile)

    components = profile["components"]
    signals = profile.get("signals", ["sled", "entropy"])

    def entropy_of(c):
        """Convergence velocity DeltaH (stored as ``delta_h`` or legacy ``entropy``)."""
        return c.get("delta_h", c.get("entropy"))

    signal_scores = {}
    if "sled" in signals:
        signal_scores["sled"] = {c["id"]: c["sled"] for c in components}
    if "entropy" in signals:
        signal_scores["entropy"] = {c["id"]: entropy_of(c) for c in components}
    if not signal_scores:
        raise SystemExit("Profile contains no recognizable signals (need sled and/or entropy).")
    criticality = pca_criticality(signal_scores)
    delta_raw = {c["id"]: entropy_of(c) for c in components}
    delta_h_hat = sigmoid(zscore(delta_raw)) if delta_raw else {}

    print("\nLoading model and dataset...")
    quantizer = TargetedQuantizer(args.model_name)
    num_layers = quantizer.num_layers
    params = estimate_component_params(quantizer.model, num_layers)

    outlier_share = estimate_actual_outlier_share(quantizer.model, num_layers, outlier_channels)
    allocator = MpcAllocator(criticality, params, delta_h_hat, outlier_share=outlier_share,
                             low3_max_frac=args.low3_max_frac, low6_max_frac=args.low6_max_frac)

    if args.target_densities:
        targets = [float(x) for x in args.target_densities.split(",")]
    else:
        targets = list(np.arange(min(BITS), 16.0 + args.bits_step, args.bits_step))
        if not any(abs(t - args.bits_per_param) < 1e-9 for t in targets):
            targets = sorted(targets + [args.bits_per_param])
    if args.bits_per_param < min(BITS):
        raise SystemExit(f"--bits-per-param must be >= {min(BITS)}")

    print(f"\nMCKP frontier: {len(targets)} density targets, "
          f"outlier_share={outlier_share:.5f} ({len(outlier_channels)} comps protected)")
    ds = load_dataset("gsm8k", "main", split="test")
    eval_dataset = list(ds)[:args.samples]

    print("\nEvaluating Baseline (BF16)...")
    baseline_acc = evaluate_gsm8k(quantizer.model, quantizer.tokenizer, eval_dataset)
    baseline_size = allocator.p_total * 2.0  # bytes: 16-bit = 2 bytes/param
    print(f"Baseline Target Prob: {baseline_acc:.4f} | size: {baseline_size / 1e6:.2f} MB")

    accuracy_floor = baseline_acc * args.min_accuracy_floor

    frontier_results = []
    seen = set()
    for cand in allocator.pareto_frontier(targets):
        cfg = cand["config"]
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        quantizer.quantize_components(cfg, outlier_channels=outlier_channels)
        acc = evaluate_gsm8k(quantizer.model, quantizer.tokenizer, eval_dataset)
        r = dict(cand)
        r["accuracy"] = acc
        r["component_configs"] = cfg
        frontier_results.append(r)
        tqdm.write(
            f"[2b={r['n_2bit']} 3b={r['n_3bit']} 4b={r['n_4bit']} 6b={r['n_6bit']} "
            f"8b={r['n_8bit']} 16b={r['n_16bit']}] "
            f"{r['bits_per_param']:.2f} bpw  size-> {r['size_reduction_pct']:.1f}%  acc {acc:.4f}"
        )

    if not frontier_results:
        raise SystemExit("Frontier is empty; nothing to select.")

    best = select_best(frontier_results, accuracy_floor, baseline_acc, baseline_size)

    from collections import defaultdict
    layer_votes = defaultdict(list)
    for cid, bit in best["config"].items():
        layer_str, _c = cid.rsplit(".", 1)
        layer_votes[int(layer_str)].append(bit)
    layer_configs = {l: max(bits, key=bits.count) for l, bits in layer_votes.items()}

    out_dir = os.path.join(base_dir, "quantize")
    os.makedirs(out_dir, exist_ok=True)

    output_data = {
        "model_name": args.model_name,
        "granularity": "subcomponent",
        "signals": signals,
        "criticality": {cid: round(float(v), 6) for cid, v in criticality.items()},
        "baseline_accuracy": baseline_acc,
        "baseline_size_bytes": baseline_size,
        "outlier_share": outlier_share,
        "outlier_channels_file": oc_path if os.path.exists(oc_path) else None,
        "target_bits_per_param": args.bits_per_param,
        "accuracy_floor": accuracy_floor,
        "optimal_config": {
            "accuracy": best["accuracy"],
            "baseline_accuracy": baseline_acc,
            "bits_per_param": best["bits_per_param"],
            "size_reduction_pct": best["size_reduction_pct"],
            "accuracy_drop_pct": best["accuracy_drop_pct"],
            "efficiency": best["efficiency"],
            "n_2bit": best["n_2bit"],
            "n_3bit": best["n_3bit"],
            "n_4bit": best["n_4bit"],
            "n_6bit": best["n_6bit"],
            "n_8bit": best["n_8bit"],
            "n_16bit": best["n_16bit"],
            "component_configs": best["config"],
            "layer_configs": layer_configs,
        },
        "frontier": [
            {
                "bits_per_param": r["bits_per_param"],
                "size_reduction_pct": r["size_reduction_pct"],
                "accuracy": r["accuracy"],
                "accuracy_drop_pct": r["accuracy_drop_pct"],
                "efficiency": r["efficiency"],
                "n_2bit": r["n_2bit"],
                "n_3bit": r["n_3bit"],
                "n_4bit": r["n_4bit"],
                "n_6bit": r["n_6bit"],
                "n_8bit": r["n_8bit"],
                "n_16bit": r["n_16bit"],
            }
            for r in frontier_results
        ],
    }

    json_path = os.path.join(out_dir, "optimal_mixed_precision.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved optimal config to {json_path}")

    if args.plot:
        _plot_frontier(frontier_results, best, baseline_acc,
                       os.path.join(out_dir, "frontier.png"))


if __name__ == "__main__":
    main()