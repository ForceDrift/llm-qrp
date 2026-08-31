"""Cross-model analysis visualizations (paper Section 4.6).

Reads the saved per-model artifacts (``optimal_mixed_precision.json`` for each
model in ``results/<model>/quantize``) and produces, into
``results/combined``:

  * ``criticality_heatmap.png`` - per-(layer, component) reasoning criticality
    ``R_{l,c}`` for every model, stacked as separate heatmap panels.
  * ``precision_allocation.png`` - the selected bit-width per component for each
    model, showing how precision is allocated across layers (attn vs. mlp).
  * ``cross_model_summary.json`` + ``cross_model_summary.md`` - a compact table
    per model: parameters, avg bits/param, compression ratio, and per-dataset
    best-method / best-value (non-BF16), computed from the benchmark JSONs.

No GPU required: all inputs are pre-saved results.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


MODELS = [
    ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M"),
    ("ibm-granite/granite-4.0-350m-base", "granite-4.0-350m"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("LiquidAI/LFM2-350M", "LFM2-350M"),
]

BAND_LABEL = {"2bit": 2, "3bit": 3, "4bit": 4, "6bit": 6, "8bit": 8, "16bit": 16, "bf16": 16}


def _find_opt(results_root, model_id):
    safe = model_id.replace("/", "_")
    p = os.path.join(results_root, safe, "quantize", "optimal_mixed_precision.json")
    if os.path.exists(p):
        return p, safe
    return None, None


def _find_benchmark(results_root, safe):
    p = os.path.join(results_root, safe, "quantize", "multi_dataset_benchmark.json")
    if os.path.exists(p):
        return p
    return None


def _best_per_dataset(bench_json):
    """Non-BF16 best (accuracy, method) per dataset from a benchmark JSON."""
    out = {}
    dr = bench_json.get("dataset_results", {})
    for ds_key, entry in dr.items():
        accs = entry.get("accuracy", {})
        best_acc = -1.0
        best_m = None
        for m, a in accs.items():
            if m == "bf16":
                continue
            if a > best_acc:
                best_acc, best_m = a, m
        out[ds_key] = {"best_method": best_m, "best_value": best_acc}
    return out


def _component_grid(criticality, comp_configs, num_layers):
    """Return (2xL) arrays of criticality and bit-width, attn then mlp rows."""
    cr = np.full((2, num_layers), np.nan)
    bw = np.full((2, num_layers), np.nan)
    for idx in range(num_layers):
        for row, c in enumerate(("attn", "mlp")):
            cid = f"{idx}.{c}"
            cr[row, idx] = criticality.get(cid, np.nan)
            bit = comp_configs.get(cid)
            bw[row, idx] = BAND_LABEL.get(bit, np.nan)
    return cr, bw


def gather(results_root):
    """Return per-model dict with criticality/config/benchmark summaries."""
    models = []
    used_models = []
    for model_id, display in MODELS:
        opt_path, safe = _find_opt(results_root, model_id)
        bench_path = _find_benchmark(results_root, safe) if safe else None
        if not opt_path:
            continue
        with open(opt_path) as f:
            od = json.load(f)
        opt = od["optimal_config"]
        cr = od.get("criticality", {})
        comp = opt.get("component_configs", opt.get("layer_configs", {}))
        num_layers = max((int(k.split(".")[0]) if "." in k else int(k)) for k in comp) + 1
        summary = {
            "model": display,
            "bits_per_param": opt.get("bits_per_param"),
            "size_reduction_pct": opt.get("size_reduction_pct"),
            "compression_ratio": od.get("compression_ratio"),
            "baseline_accuracy": od.get("baseline_accuracy"),
        }
        bench = None
        if bench_path:
            with open(bench_path) as f:
                bench = json.load(f)
            summary["compression_ratio"] = bench.get(
                "compression_ratio", summary["compression_ratio"])
            summary["best_per_dataset"] = _best_per_dataset(bench)
        models.append({
            "id": model_id,
            "display": display,
            "num_layers": num_layers,
            "criticality": cr,
            "component_configs": comp,
            "summary": summary,
            "benchmark": bench,
        })
        used_models.append(display)
    return models, used_models


def render_heatmap(models, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.5), squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, models):
        cr, _ = _component_grid(m["criticality"], m["component_configs"], m["num_layers"])
        ax.imshow(cr, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(m["display"], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["attn", "mlp"])
        ax.set_xlabel("Layer index")
        ax.grid(False)
    fig.suptitle("Reasoning criticality R_{l,c} across models", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_allocation(models, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.5), squeeze=False)
    axes = axes[0]
    vmax = 16
    cmap = matplotlib.colormaps.get_cmap("RdYlBu_r").resampled(vmax)
    for ax, m in zip(axes, models):
        _, bw = _component_grid(m["criticality"], m["component_configs"], m["num_layers"])
        # NaN (unlisted) components stay BF16 -> 16 bits
        masked = np.where(np.isnan(bw), 16.0, bw)
        ax.imshow(masked, aspect="auto", cmap=cmap, vmin=2, vmax=16)
        ax.set_title(f"{m['display']} ({m['summary'].get('bits_per_param', float('nan')):.2f} bpw)",
                     fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["attn", "mlp"])
        ax.set_xlabel("Layer index")
        ax.grid(False)
    fig.suptitle("Selected precision per sub-component (bits)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_summary(models, json_path, md_path):
    rows = {}
    for m in models:
        s = m["summary"]
        bpd = s.get("best_per_dataset", {})
        rows[m["display"]] = {
            "bits_per_param": round(s["bits_per_param"], 3) if s.get("bits_per_param") else None,
            "size_reduction_pct": round(s["size_reduction_pct"], 2) if s.get("size_reduction_pct") is not None else None,
            "compression_ratio": round(s["compression_ratio"], 3) if s.get("compression_ratio") else None,
            "baseline_accuracy": s.get("baseline_accuracy"),
            "best_per_dataset": {k: {"method": v["best_method"], "value": round(v["best_value"], 4)}
                                 for k, v in bpd.items()},
        }
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    with open(md_path, "w") as f:
        f.write("| Model | bpw | Size red. | Compress. | GSM8K best | TruthfulQA best | MMLU best |\n")
        f.write("|-------|-----|-----------|-----------|------------|-----------------|-----------|\n")
        for name, r in rows.items():
            bpd = r["best_per_dataset"]
            cells = [name,
                     f"{r['bits_per_param']:.2f}" if r["bits_per_param"] else "-",
                     f"{r['size_reduction_pct']:.1f}%" if r["size_reduction_pct"] is not None else "-",
                     f"{r['compression_ratio']:.2f}x" if r["compression_ratio"] else "-",
                     f"{bpd.get('gsm8k',{}).get('method','-')} {bpd.get('gsm8k',{}).get('value','-'):g}" if "gsm8k" in bpd else "-",
                     f"{bpd.get('tfqa',{}).get('method','-')} {bpd.get('tfqa',{}).get('value','-'):g}" if "tfqa" in bpd else "-",
                     f"{bpd.get('mmlu',{}).get('method','-')} {bpd.get('mmlu',{}).get('value','-'):g}" if "mmlu" in bpd else "-"]
            f.write("| " + " | ".join(cells) + " |\n")


def main():
    ap = argparse.ArgumentParser(description="Cross-model analysis figures")
    ap.add_argument("--results-root", type=str, required=True,
                    help="Base results directory containing per-model folders")
    ap.add_argument("--output-folder", type=str, default=None,
                    help="Where figures go (default: <results-root>/combined)")
    args = ap.parse_args()

    root = args.results_root
    out_dir = args.output_folder or os.path.join(root, "combined")
    os.makedirs(out_dir, exist_ok=True)

    models, used = gather(root)
    if not models:
        raise SystemExit("No optimal_mixed_precision.json found for any model. Run allocation first.")
    print("Models found:", used)

    heatmap_path = os.path.join(out_dir, "criticality_heatmap.png")
    alloc_path = os.path.join(out_dir, "precision_allocation.png")
    summary_json = os.path.join(out_dir, "cross_model_summary.json")
    summary_md = os.path.join(out_dir, "cross_model_summary.md")

    render_heatmap(models, heatmap_path)
    print("Saved", heatmap_path)
    render_allocation(models, alloc_path)
    print("Saved", alloc_path)
    render_summary(models, summary_json, summary_md)
    print("Saved", summary_json)
    print("Saved", summary_md)


if __name__ == "__main__":
    main()
