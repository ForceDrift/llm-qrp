"""
thinking_ablation_benchmark.py — Phase 1 ablation benchmark.

Uses precomputed layer importance scores (from aggregate_scores.py) to
identify "thinking" layers, then measures GSM8K accuracy under three conditions:

  - baseline:   no layers ablated
  - top20%:     top 20% of layers by score ablated (highest importance)
  - bottom20%:  bottom 20% of layers by score ablated (lowest importance)

Expected outcome: ablating top-20% should cause a larger accuracy drop than
ablating bottom-20%, proving those layers encode reasoning / "thinking".

Usage:
    python -m qrp.benchmark.thinking_ablation_benchmark \\
        --scores-file results/layer_avg_scores.json \\
        --model-name HuggingFaceTB/SmolLM2-360M \\
        --dataset gsm8k \\
        --limit 200 \\
        --output-folder results/ablation/
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset

from qrp.analysis.ablation_controller import AblationController
from qrp.analysis.aggregate_scores import getSortedLayers
from qrp.benchmark.evaluate_gsm8k import evaluate_gsm8k


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_samples(dataset_name: str, limit: int) -> list:
    """Return a list of {"question": ..., "answer": ...} dicts."""
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [{"question": x["question"], "answer": x["answer"]} for x in ds][:limit]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Only 'gsm8k' is supported for ablation.")


def load_sorted_layers(scores_file: str) -> list:
    """Load sorted layer list from aggregate_scores output file.

    Returns list of (layer_idx, score) tuples sorted ascending by score.
    """
    with open(scores_file, "r") as f:
        data = json.load(f)

    # Support both raw avgScores dict and the full aggregate_scores output format
    if "sortedAscending" in data:
        return [(int(entry[0]), float(entry[1])) for entry in data["sortedAscending"]]
    elif "layerAvgScores" in data:
        return getSortedLayers(data["layerAvgScores"])
    else:
        # Assume it's just a flat dict of layer_idx → score
        return getSortedLayers(data)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    model_name: str,
    scores_file: str,
    dataset_name: str,
    limit: int,
    fraction: float,
    output_folder: str,
    verbose: bool = False,
):
    """Run the three-condition ablation benchmark and save results."""

    print(f"\n{'='*60}")
    print(f"Thinking Ablation Benchmark")
    print(f"  Model:    {model_name}")
    print(f"  Dataset:  {dataset_name} (n={limit})")
    print(f"  Fraction: {fraction*100:.0f}%")
    print(f"  Scores:   {scores_file}")
    print(f"{'='*60}\n")

    # Load samples and sorted layer scores
    samples = load_dataset_samples(dataset_name, limit)
    sorted_layers = load_sorted_layers(scores_file)
    print(f"Loaded {len(samples)} samples, {len(sorted_layers)} scored layers.\n")

    # Load model once via AblationController (handles hook management)
    ctrl = AblationController(model_name)

    all_results = {}

    # ------------------------------------------------------------------
    # Condition 1: Baseline (no ablation)
    # ------------------------------------------------------------------
    print("--- Running: baseline ---")
    ctrl.restore_layers()
    t0 = time.time()
    acc_base, details_base = evaluate_gsm8k(
        ctrl.model, ctrl.tokenizer, samples, device=ctrl.device, verbose=verbose
    )
    elapsed = time.time() - t0
    print(f"  Accuracy: {acc_base:.4f}  ({elapsed:.1f}s)\n")
    all_results["baseline"] = {
        "layers_ablated": [],
        "num_layers_ablated": 0,
        "accuracy": acc_base,
        "elapsed_sec": round(elapsed, 1),
        "details": details_base,
    }

    # ------------------------------------------------------------------
    # Condition 2: Ablate top fraction (highest importance → "thinking")
    # ------------------------------------------------------------------
    print(f"--- Running: top {fraction*100:.0f}% ablated ---")
    top_layers = ctrl.ablate_top_fraction(sorted_layers, fraction=fraction)
    print(f"  Ablated layers: {sorted(top_layers)}")
    t0 = time.time()
    acc_top, details_top = evaluate_gsm8k(
        ctrl.model, ctrl.tokenizer, samples, device=ctrl.device, verbose=verbose
    )
    elapsed = time.time() - t0
    drop_top = acc_base - acc_top
    print(f"  Accuracy: {acc_top:.4f}  drop={drop_top:+.4f}  ({elapsed:.1f}s)\n")
    all_results[f"top_{int(fraction*100)}pct"] = {
        "layers_ablated": sorted(top_layers),
        "num_layers_ablated": len(top_layers),
        "accuracy": acc_top,
        "accuracy_drop": drop_top,
        "elapsed_sec": round(elapsed, 1),
        "details": details_top,
    }

    # ------------------------------------------------------------------
    # Condition 3: Ablate bottom fraction (lowest importance → "noise")
    # ------------------------------------------------------------------
    print(f"--- Running: bottom {fraction*100:.0f}% ablated ---")
    bottom_layers = ctrl.ablate_bottom_fraction(sorted_layers, fraction=fraction)
    print(f"  Ablated layers: {sorted(bottom_layers)}")
    t0 = time.time()
    acc_bot, details_bot = evaluate_gsm8k(
        ctrl.model, ctrl.tokenizer, samples, device=ctrl.device, verbose=verbose
    )
    elapsed = time.time() - t0
    drop_bot = acc_base - acc_bot
    print(f"  Accuracy: {acc_bot:.4f}  drop={drop_bot:+.4f}  ({elapsed:.1f}s)\n")
    all_results[f"bottom_{int(fraction*100)}pct"] = {
        "layers_ablated": sorted(bottom_layers),
        "num_layers_ablated": len(bottom_layers),
        "accuracy": acc_bot,
        "accuracy_drop": drop_bot,
        "elapsed_sec": round(elapsed, 1),
        "details": details_bot,
    }

    ctrl.restore_layers()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"  Baseline accuracy:             {acc_base:.4f}")
    print(f"  Top-{int(fraction*100)}% ablated accuracy:     {acc_top:.4f}  (drop={drop_top:+.4f})")
    print(f"  Bottom-{int(fraction*100)}% ablated accuracy:  {acc_bot:.4f}  (drop={drop_bot:+.4f})")
    if drop_top > drop_bot:
        print(f"\n  ✓ TOP layers cause larger accuracy drop → confirmed as 'thinking' layers")
    else:
        print(f"\n  ✗ Bottom layers cause equal or larger drop (unexpected — check scores)")
    print("="*60 + "\n")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(output_folder, exist_ok=True)
    out_file = os.path.join(output_folder, "ablation_results.json")

    output = {
        "config": {
            "model_name": model_name,
            "dataset": dataset_name,
            "limit": limit,
            "fraction": fraction,
            "scores_file": scores_file,
        },
        "sorted_layers": sorted_layers,
        "results": all_results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {out_file}")
    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thinking Ablation Benchmark")
    parser.add_argument(
        "--scores-file", type=str, required=True,
        help="Path to layer_avg_scores.json (output of aggregate_scores.py)"
    )
    parser.add_argument(
        "--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        help="Dataset to evaluate on (currently only gsm8k supported)"
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--fraction", type=float, default=0.2,
        help="Fraction of layers to ablate in each condition (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--output-folder", type=str, required=True,
        help="Folder to save ablation_results.json"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample results"
    )

    args = parser.parse_args()

    run_benchmark(
        model_name=args.model_name,
        scores_file=args.scores_file,
        dataset_name=args.dataset,
        limit=args.limit,
        fraction=args.fraction,
        output_folder=args.output_folder,
        verbose=args.verbose,
    )
