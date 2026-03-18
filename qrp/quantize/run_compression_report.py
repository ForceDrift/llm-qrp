"""
Final compression benchmark report.
Reads optimal_mixed_precision.json and quantize_sweep_results.json and produces
a printed summary + a comparison chart (compression_benchmark.png).
"""

import os
import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def bytes_to_mb(b):
    return b / 1e6


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def bits_per_param(precision):
    return {"bf16": 16, "8bit": 8, "4bit": 4}.get(precision, 16)


def print_section(title):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Produce a final compression benchmark report comparing "
                    "BF16, sweep configs, and the optimal mixed-precision model."
    )
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output-folder", type=str, required=True)
    args = parser.parse_args()

    model_safe = args.model_name.replace("/", "_")
    base_dir = os.path.join(args.output_folder, model_safe)
    quantize_dir = os.path.join(base_dir, "quantize")

    # --- Load data ---
    sweep_data = load_json(os.path.join(quantize_dir, "quantize_sweep_results.json"))
    opt_data = load_json(os.path.join(quantize_dir, "optimal_mixed_precision.json"))

    if opt_data is None:
        raise FileNotFoundError(
            f"Could not find optimal_mixed_precision.json in {quantize_dir}.\n"
            "Run find_optimal_mixed_precision.py first."
        )

    baseline_acc = opt_data["baseline_accuracy"]
    baseline_size = opt_data["baseline_size_bytes"]
    optimal = opt_data["optimal_config"]

    # Build comparison table rows
    rows = []

    # Baseline
    rows.append({
        "label": "BF16 (Baseline)",
        "precision": "BF16",
        "accuracy": baseline_acc,
        "size_mb": bytes_to_mb(baseline_size),
        "size_reduction_pct": 0.0,
        "acc_drop_pct": 0.0,
        "compression_ratio": 1.0,
    })

    # Sweep results
    if sweep_data:
        for pct, acc in sweep_data.get("performance_8bit", []):
            # Estimate size: all quantized layers at 8-bit → ~50% of param bytes
            # We approximate from known sizes in opt_data candidates if possible
            size_reduction = (pct / 100.0) * 50.0  # rough: pct% of layers at 8-bit = ~0.5x savings on those
            rows.append({
                "label": f"8-bit ({pct}% layers)",
                "precision": "INT8",
                "accuracy": acc,
                "size_mb": bytes_to_mb(baseline_size) * (1 - size_reduction / 100),
                "size_reduction_pct": size_reduction,
                "acc_drop_pct": (1 - acc / baseline_acc) * 100 if baseline_acc > 0 else 0,
                "compression_ratio": 1 / (1 - size_reduction / 100) if size_reduction < 100 else float('inf'),
            })

        for pct, acc in sweep_data.get("performance_4bit", []):
            size_reduction = (pct / 100.0) * 75.0  # 4-bit = 75% savings on those layers
            rows.append({
                "label": f"4-bit ({pct}% layers)",
                "precision": "FP4",
                "accuracy": acc,
                "size_mb": bytes_to_mb(baseline_size) * (1 - size_reduction / 100),
                "size_reduction_pct": size_reduction,
                "acc_drop_pct": (1 - acc / baseline_acc) * 100 if baseline_acc > 0 else 0,
                "compression_ratio": 1 / (1 - size_reduction / 100) if size_reduction < 100 else float('inf'),
            })

    # Optimal mixed
    opt_size = optimal["estimated_size_bytes"]
    opt_acc = optimal["accuracy"]
    opt_size_reduction = optimal["size_reduction_pct"]
    opt_acc_drop = optimal["accuracy_drop_pct"]
    n4 = optimal["n_4bit"]
    n8 = optimal["n_8bit_only"]
    n_bf = optimal.get("n_bf16", 0)
    compression_ratio = baseline_size / opt_size if opt_size > 0 else 1.0

    rows.append({
        "label": f"Optimal Mixed ({n4}L FP4 + {n8}L INT8 + {n_bf}L BF16)",
        "precision": "MIXED",
        "accuracy": opt_acc,
        "size_mb": bytes_to_mb(opt_size),
        "size_reduction_pct": opt_size_reduction,
        "acc_drop_pct": opt_acc_drop,
        "compression_ratio": compression_ratio,
    })

    # --- Print summary table ---
    print_section(f"COMPRESSION BENCHMARK — {args.model_name}")
    print(f"{'Config':<40} {'Acc':>7} {'Acc Drop':>9} {'Size (MB)':>10} {'Size ↓':>8} {'Ratio':>7}")
    print("-" * 85)

    for r in rows:
        marker = " ◀ OPTIMAL" if r["label"].startswith("Optimal") else ""
        print(
            f"{r['label']:<40} "
            f"{r['accuracy']:>7.4f} "
            f"{r['acc_drop_pct']:>8.1f}% "
            f"{r['size_mb']:>9.2f} "
            f"{r['size_reduction_pct']:>7.1f}% "
            f"{r['compression_ratio']:>6.2f}x"
            f"{marker}"
        )

    print_section("OPTIMAL CONFIG SUMMARY")
    print(f"  Model:              {args.model_name}")
    print(f"  Baseline Size:      {bytes_to_mb(baseline_size):.2f} MB")
    print(f"  Optimal Size:       {bytes_to_mb(opt_size):.2f} MB")
    print(f"  Size Reduction:     {opt_size_reduction:.1f}%")
    print(f"  Compression Ratio:  {compression_ratio:.2f}x")
    print(f"  Baseline Accuracy:  {baseline_acc:.4f}")
    print(f"  Optimal Accuracy:   {opt_acc:.4f}")
    print(f"  Accuracy Retained:  {(1 - opt_acc_drop / 100) * 100:.1f}%")
    print(f"  Accuracy Drop:      {opt_acc_drop:.2f}%")
    print(f"  Efficiency Score:   {optimal['efficiency']:.2f}")
    print(f"  Layer config:       {n4} × FP4  |  {n8} × INT8  |  {n_bf} × BF16")

    # --- Generate comparison chart ---
    _generate_chart(rows, baseline_acc, baseline_size, args.model_name, quantize_dir)
    print(f"\nSaved report chart to {os.path.join(quantize_dir, 'compression_benchmark.png')}")


def _generate_chart(rows, baseline_acc, baseline_size, model_name, out_dir):
    color_map = {
        "BF16": "#607D8B",
        "INT8": "#42A5F5",
        "FP4": "#EF5350",
        "MIXED": "#66BB6A",
    }

    # Filter to a clean set for bars — exclude redundant baseline copies
    display_rows = rows

    labels = [r["label"] for r in display_rows]
    accs = [r["accuracy"] for r in display_rows]
    sizes = [r["size_mb"] for r in display_rows]
    reductions = [r["size_reduction_pct"] for r in display_rows]
    colors = [color_map[r["precision"]] for r in display_rows]

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#f9f9f9")
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # Accuracy bar — full width
    ax2 = fig.add_subplot(gs[1, 0])   # Size bar
    ax3 = fig.add_subplot(gs[1, 1])   # Size reduction bar

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.65

    # ── Accuracy comparison ──────────────────────────────────────
    bars1 = ax1.bar(x, accs, width=bar_w, color=colors, edgecolor="#333", linewidth=0.6, alpha=0.9)
    ax1.axhline(baseline_acc, color="#607D8B", linestyle=":", linewidth=1.4, alpha=0.7)
    ax1.set_title("Target Probability (Accuracy)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Target Prob  exp(−loss)")
    ax1.set_ylim(min(accs) * 0.97, max(accs) * 1.03)
    ax1.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(accs) * 0.003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Highlight optimal bar
    opt_idx = next((i for i, r in enumerate(display_rows) if r["precision"] == "MIXED"), None)
    if opt_idx is not None:
        bars1[opt_idx].set_edgecolor("#004D40")
        bars1[opt_idx].set_linewidth(2.5)

    # ── Model size comparison ─────────────────────────────────────
    bars2 = ax2.bar(x, sizes, width=bar_w, color=colors, edgecolor="#333", linewidth=0.6, alpha=0.9)
    ax2.set_title("Estimated Layer Size (MB)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
    ax2.set_ylabel("Size (MB)")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.01,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    if opt_idx is not None:
        bars2[opt_idx].set_edgecolor("#004D40")
        bars2[opt_idx].set_linewidth(2.5)

    # ── Size reduction % ──────────────────────────────────────────
    bars3 = ax3.bar(x, reductions, width=bar_w, color=colors, edgecolor="#333", linewidth=0.6, alpha=0.9)
    ax3.set_title("Size Reduction (%)", fontsize=10, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
    ax3.set_ylabel("Reduction (%)")
    ax3.set_ylim(0, max(reductions) * 1.18 if max(reductions) > 0 else 10)
    ax3.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, reductions):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(reductions) * 0.01,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    if opt_idx is not None:
        bars3[opt_idx].set_edgecolor("#004D40")
        bars3[opt_idx].set_linewidth(2.5)

    # ── Legend ────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#607D8B", label="BF16 (Full Precision)"),
        mpatches.Patch(color="#42A5F5", label="INT8 (8-bit)"),
        mpatches.Patch(color="#EF5350", label="FP4 (4-bit)"),
        mpatches.Patch(color="#66BB6A", label="Optimal Mixed ◀"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8.5,
               framealpha=0.9, edgecolor="#ccc", ncol=2)

    fig.suptitle(f"Compression Benchmark — {model_name}", fontsize=13,
                 fontweight="bold", y=1.01)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "compression_benchmark.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


if __name__ == "__main__":
    main()
