"""
Multi-dataset benchmark: BF16 baseline vs. optimal mixed-precision model.

Evaluates both models on each requested dataset and reports:
  - Target probability (exp(-avg_loss)) per dataset
  - Estimated model size (bytes)
  - Size-to-accuracy efficiency ratio (1 / (size_MB * acc_drop_frac + eps))
  - Generates a multi-panel comparison chart: benchmark_comparison.png
"""

import os
import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm

from qrp.quantize.quantizer import TargetedQuantizer


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "gsm8k": {
        "hf_path": ("gsm8k", "main"),
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
        "display": "GSM8K",
    },
    "tfqa": {
        "hf_path": ("truthful_qa", "generation"),
        "split": "validation",
        "question_key": "question",
        "answer_key": "best_answer",
        "display": "TruthfulQA",
    },
    "mmlu": {
        "hf_path": ("cais/mmlu", "all"),
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",   # int (0-3) — converted to str below
        "display": "MMLU",
    },
}


def load_eval_pairs(dataset_key, n_samples):
    """Return list of (question_str, answer_str) pairs."""
    cfg = DATASET_REGISTRY[dataset_key]
    ds = load_dataset(*cfg["hf_path"], split=cfg["split"])
    pairs = []
    for item in list(ds)[:n_samples]:
        q = item[cfg["question_key"]]
        a = item[cfg["answer_key"]]
        if dataset_key == "mmlu":
            # answer is index (0-3); map to the option text
            choices = item.get("choices", [])
            a = choices[int(a)] if choices else str(a)
        pairs.append((str(q), str(a)))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(model, tokenizer, pairs, dataset_key):
    """
    Compute target probability exp(-avg_cross_entropy_loss) over (prompt, answer) pairs.
    Uses a chain-of-thought prompt style consistent with the existing codebase.
    """
    model.eval()
    total_loss = 0.0
    valid = 0

    for question, answer in tqdm(pairs, desc=f"  Evaluating {dataset_key}", leave=False):
        if dataset_key == "gsm8k":
            prompt = f"Question: {question}\nAnswer: Let's think step by step\n"
        elif dataset_key == "tfqa":
            prompt = f"Question: {question}\nAnswer: "
        elif dataset_key == "mmlu":
            prompt = f"Question: {question}\nAnswer: "
        else:
            prompt = f"Question: {question}\nAnswer: "

        prompt_ids = tokenizer.encode(prompt)
        target_ids = tokenizer.encode(answer, add_special_tokens=False)

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
    avg_loss = total_loss / valid
    return math.exp(-avg_loss)


# ──────────────────────────────────────────────────────────────────────────────
# Size estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_size(model, layer_configs, num_layers):
    """Estimate total quantized-layer parameter size in bytes."""
    bytes_per_param = {"bf16": 2.0, "8bit": 1.0, "4bit": 0.5}
    attn_projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_projs = ["gate_proj", "up_proj", "down_proj"]
    total = 0.0
    for idx in range(num_layers):
        layer = model.model.layers[idx]
        params = sum(
            getattr(layer.self_attn, n).weight.numel()
            for n in attn_projs
            if hasattr(layer.self_attn, n)
        ) + sum(
            getattr(layer.mlp, n).weight.numel()
            for n in mlp_projs
            if hasattr(layer.mlp, n)
        )
        precision = layer_configs.get(idx, "bf16")
        total += params * bytes_per_param[precision]
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def generate_chart(results, baseline_size_mb, opt_size_mb, model_name, out_path):
    """
    results: dict keyed by dataset_key ->
        {"display": str, "baseline_acc": float, "opt_acc": float}
    """
    datasets = list(results.keys())
    displays = [results[d]["display"] for d in datasets]
    baseline_accs = [results[d]["baseline_acc"] for d in datasets]
    opt_accs = [results[d]["opt_acc"] for d in datasets]

    # Efficiency = accuracy / size_MB  (higher is better)
    baseline_eff = [a / baseline_size_mb for a in baseline_accs]
    opt_eff = [a / opt_size_mb for a in opt_accs]

    x = np.arange(len(datasets))
    w = 0.35
    BLU = "#546E7A"
    GRN = "#43A047"

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#fafafa")
    gs = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.38)

    ax_acc   = fig.add_subplot(gs[0, :])   # accuracy — full width
    ax_size  = fig.add_subplot(gs[1, 0])   # model size
    ax_eff   = fig.add_subplot(gs[1, 1])   # efficiency

    # ── Accuracy ──────────────────────────────────────────────────
    b1 = ax_acc.bar(x - w/2, baseline_accs, w, label="BF16 (Baseline)",
                    color=BLU, edgecolor="#333", linewidth=0.7, alpha=0.88)
    b2 = ax_acc.bar(x + w/2, opt_accs, w, label="Optimal Mixed-Precision",
                    color=GRN, edgecolor="#333", linewidth=0.7, alpha=0.88)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2, h + max(baseline_accs)*0.008,
                        f"{h:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax_acc.set_title("Target Probability  exp(−loss)  by Dataset", fontsize=12, fontweight="bold")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(displays, fontsize=11)
    ax_acc.set_ylabel("Target Prob")
    ax_acc.set_ylim(0, max(max(baseline_accs), max(opt_accs)) * 1.18)
    ax_acc.legend(fontsize=10)
    ax_acc.grid(axis="y", alpha=0.3)

    # ── Model size (single bar pair, same for all datasets) ────────
    sizes_mb = [baseline_size_mb, opt_size_mb]
    size_colors = [BLU, GRN]
    size_labels = ["BF16", "Optimal Mixed"]
    b3 = ax_size.bar(size_labels, sizes_mb, color=size_colors,
                     edgecolor="#333", linewidth=0.8, alpha=0.88, width=0.45)
    for bar, val in zip(b3, sizes_mb):
        ax_size.text(bar.get_x() + bar.get_width()/2, val + max(sizes_mb)*0.01,
                     f"{val:.1f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    reduction = (1 - opt_size_mb / baseline_size_mb) * 100
    ax_size.set_title(f"Estimated Layer Size\n({reduction:.1f}% reduction)", fontsize=10, fontweight="bold")
    ax_size.set_ylabel("Size (MB)")
    ax_size.set_ylim(0, max(sizes_mb) * 1.22)
    ax_size.grid(axis="y", alpha=0.3)

    # ── Efficiency: accuracy / size_MB ─────────────────────────────
    b4 = ax_eff.bar(x - w/2, baseline_eff, w, label="BF16",
                    color=BLU, edgecolor="#333", linewidth=0.7, alpha=0.88)
    b5 = ax_eff.bar(x + w/2, opt_eff, w, label="Mixed",
                    color=GRN, edgecolor="#333", linewidth=0.7, alpha=0.88)
    for bars in (b4, b5):
        for bar in bars:
            h = bar.get_height()
            ax_eff.text(bar.get_x() + bar.get_width()/2,
                        h + max(max(baseline_eff), max(opt_eff))*0.012,
                        f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    ax_eff.set_title("Efficiency  (Accuracy / Size_MB)\nHigher = better per MB", fontsize=10, fontweight="bold")
    ax_eff.set_xticks(x)
    ax_eff.set_xticklabels(displays, fontsize=10)
    ax_eff.set_ylabel("Accuracy / MB")
    ax_eff.set_ylim(0, max(max(baseline_eff), max(opt_eff)) * 1.22)
    ax_eff.legend(fontsize=9)
    ax_eff.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"BF16  vs.  Optimal Mixed-Precision Model\n{model_name}",
        fontsize=13, fontweight="bold", y=1.01
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved benchmark chart → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset benchmark: BF16 vs optimal mixed-precision model"
    )
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Base folder (same as used in all other scripts)")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples per dataset")
    parser.add_argument("--datasets", type=str, default="gsm8k,tfqa",
                        help="Comma-separated list of datasets: gsm8k, tfqa, mmlu")
    args = parser.parse_args()

    dataset_keys = [d.strip() for d in args.datasets.split(",")]
    for k in dataset_keys:
        if k not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset '{k}'. Choose from: {list(DATASET_REGISTRY)}")

    model_safe = args.model_name.replace("/", "_")
    base_dir = os.path.join(args.output_folder, model_safe)
    quantize_dir = os.path.join(base_dir, "quantize")
    opt_json = os.path.join(quantize_dir, "optimal_mixed_precision.json")

    if not os.path.exists(opt_json):
        raise FileNotFoundError(
            f"Could not find {opt_json}.\n"
            "Run find_optimal_mixed_precision.py first."
        )

    with open(opt_json) as f:
        opt_data = json.load(f)

    optimal = opt_data["optimal_config"]
    opt_layer_configs = {int(k): v for k, v in optimal["layer_configs"].items()}

    print(f"\n{'='*65}")
    print(f"  MULTI-DATASET BENCHMARK")
    print(f"  Model:    {args.model_name}")
    print(f"  Datasets: {', '.join(dataset_keys)}")
    print(f"  Samples:  {args.samples} per dataset")
    print(f"  Optimal config: {optimal['n_4bit']} × FP4  |  {optimal['n_8bit_only']} × INT8  |  {optimal.get('n_bf16', '?')} × BF16")
    print(f"{'='*65}\n")

    # Load model once
    print("Loading model (BF16 baseline)...")
    quantizer = TargetedQuantizer(args.model_name)
    num_layers = quantizer.num_layers

    baseline_size = estimate_size(quantizer.model, {}, num_layers)
    baseline_size_mb = baseline_size / 1e6
    print(f"Baseline size: {baseline_size_mb:.2f} MB")

    # Apply and measure optimal quantization size
    quantizer.quantize_layers(opt_layer_configs)
    opt_size = estimate_size(quantizer.model, opt_layer_configs, num_layers)
    opt_size_mb = opt_size / 1e6
    print(f"Optimal mixed size: {opt_size_mb:.2f} MB  ({(1 - opt_size_mb/baseline_size_mb)*100:.1f}% reduction)")

    # Restore to baseline for evaluation loop
    quantizer.restore()

    results = {}
    all_rows = []

    for ds_key in dataset_keys:
        display = DATASET_REGISTRY[ds_key]["display"]
        print(f"\n──── {display} ────")

        pairs = load_eval_pairs(ds_key, args.samples)

        # Baseline
        print("  [1/2] BF16 baseline...")
        quantizer.restore()
        baseline_acc = evaluate_dataset(
            quantizer.model, quantizer.tokenizer, pairs, ds_key
        )
        print(f"  BF16 accuracy: {baseline_acc:.4f}")

        # Optimal quantized
        print("  [2/2] Optimal mixed-precision...")
        quantizer.quantize_layers(opt_layer_configs)
        opt_acc = evaluate_dataset(
            quantizer.model, quantizer.tokenizer, pairs, ds_key
        )
        print(f"  Mixed accuracy: {opt_acc:.4f}")
        quantizer.restore()

        acc_drop = (1 - opt_acc / baseline_acc) * 100 if baseline_acc > 0 else 0
        baseline_eff = baseline_acc / baseline_size_mb
        opt_eff = opt_acc / opt_size_mb
        eff_gain = (opt_eff / baseline_eff - 1) * 100 if baseline_eff > 0 else 0

        results[ds_key] = {
            "display": display,
            "baseline_acc": baseline_acc,
            "opt_acc": opt_acc,
            "acc_drop_pct": acc_drop,
            "baseline_eff": baseline_eff,
            "opt_eff": opt_eff,
            "eff_gain_pct": eff_gain,
        }

        all_rows.append({
            "dataset": display,
            "model": "BF16",
            "accuracy": baseline_acc,
            "size_mb": baseline_size_mb,
            "efficiency": baseline_eff,
        })
        all_rows.append({
            "dataset": display,
            "model": "Optimal Mixed",
            "accuracy": opt_acc,
            "size_mb": opt_size_mb,
            "efficiency": opt_eff,
        })

    # ── Print final table ──────────────────────────────────────────
    col_w = 14
    print(f"\n{'='*80}")
    print("  FINAL BENCHMARK RESULTS")
    print(f"{'='*80}")
    header = (
        f"{'Dataset':<12} {'Model':<18} {'Accuracy':>{col_w}} "
        f"{'Size (MB)':>{col_w}} {'Acc Drop':>{col_w}} {'Eff (Acc/MB)':>{col_w}} {'Eff Gain':>{col_w}}"
    )
    print(header)
    print("-" * 80)

    for ds_key, r in results.items():
        display = r["display"]
        # BF16 row
        print(
            f"{display:<12} {'BF16':<18} {r['baseline_acc']:>{col_w}.4f} "
            f"{baseline_size_mb:>{col_w}.2f} {'—':>{col_w}} {r['baseline_eff']:>{col_w}.6f} {'—':>{col_w}}"
        )
        # Optimal row
        eff_str = f"+{r['eff_gain_pct']:.1f}%" if r['eff_gain_pct'] >= 0 else f"{r['eff_gain_pct']:.1f}%"
        print(
            f"{'':12} {'Optimal Mixed':<18} {r['opt_acc']:>{col_w}.4f} "
            f"{opt_size_mb:>{col_w}.2f} {r['acc_drop_pct']:>{col_w - 1}.1f}% {r['opt_eff']:>{col_w}.6f} {eff_str:>{col_w}}"
        )
        print("-" * 80)

    size_reduction_pct = (1 - opt_size_mb / baseline_size_mb) * 100
    print(f"\n  Compression:  {baseline_size_mb:.2f} MB  →  {opt_size_mb:.2f} MB  ({size_reduction_pct:.1f}% reduction, {baseline_size_mb/opt_size_mb:.2f}x smaller)")
    print(f"  Optimal config: {optimal['n_4bit']} × FP4  |  {optimal['n_8bit_only']} × INT8  |  {optimal.get('n_bf16', '?')} × BF16")

    # ── Save JSON ──────────────────────────────────────────────────
    out_json = os.path.join(quantize_dir, "multi_dataset_benchmark.json")
    with open(out_json, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "samples_per_dataset": args.samples,
            "baseline_size_mb": baseline_size_mb,
            "optimal_size_mb": opt_size_mb,
            "size_reduction_pct": size_reduction_pct,
            "compression_ratio": baseline_size_mb / opt_size_mb,
            "optimal_layer_config": optimal,
            "dataset_results": {k: {
                "baseline_accuracy": v["baseline_acc"],
                "optimal_accuracy": v["opt_acc"],
                "accuracy_drop_pct": v["acc_drop_pct"],
                "baseline_efficiency": v["baseline_eff"],
                "optimal_efficiency": v["opt_eff"],
                "efficiency_gain_pct": v["eff_gain_pct"],
            } for k, v in results.items()},
        }, f, indent=2)
    print(f"\n  Saved JSON → {out_json}")

    # ── Generate chart ─────────────────────────────────────────────
    chart_path = os.path.join(quantize_dir, "benchmark_comparison.png")
    generate_chart(results, baseline_size_mb, opt_size_mb, args.model_name, chart_path)


if __name__ == "__main__":
    main()
