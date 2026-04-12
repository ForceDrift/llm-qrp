"""
GPTQ Mixed-Precision Benchmark: Compare quantization approaches.

Evaluates three configurations:
  1. BF16 baseline (no quantization)
  2. Uniform GPTQ 4-bit (all layers quantized — what AWQ/GPTQ papers do)
  3. QRP + GPTQ mixed (important layers stay bf16, rest get GPTQ 4-bit)

Benchmarks on:
  - WikiText-2 perplexity (standard quantization metric)
  - GSM8K exact-match accuracy (reasoning preservation metric)
  - Target probability (QRP's existing metric, for reference)

Usage:
  python -m qrp.quantize.run_gptq_benchmark \\
      --model-name HuggingFaceTB/SmolLM2-135M \\
      --output-folder ./results \\
      --samples 1
"""

import argparse
import csv
import json
import os

import torch

from qrp.quantize.gptq_core import GPTQMixedQuantizer
from qrp.benchmark.eval_standard import (
    evaluate_wikitext2_ppl,
    evaluate_gsm8k_exact_match,
    evaluate_target_prob,
)


def load_qrp_precision_map(aggregated_file, num_layers, pct_bf16=30, pct_8bit=0):
    """
    Convert QRP's aggregated layer scores into a precision map.

    Layers with the highest "thinking" scores stay at bf16.
    Layers with medium scores get 8-bit.
    Layers with the lowest scores get 4-bit.

    Args:
        aggregated_file: path to aggregated_scores.json
        num_layers: total number of layers in the model
        pct_bf16: percentage of layers to keep at bf16 (top scoring)
        pct_8bit: percentage of layers to assign 8-bit (middle scoring)

    Returns:
        dict mapping layer_idx -> "4bit" | "8bit" | "bf16"
    """
    with open(aggregated_file, "r") as f:
        scores = json.load(f)

    sorted_layers = sorted(
        [(int(key.split("_")[1]), val) for key, val in scores.items()],
        key=lambda x: x[1]
    )

    n_bf16 = max(1, int(num_layers * pct_bf16 / 100))
    n_8bit = int(num_layers * pct_8bit / 100)

    configs = {}
    for rank, (layer_idx, score) in enumerate(sorted_layers):
        if rank < len(sorted_layers) - n_bf16 - n_8bit:
            configs[layer_idx] = "4bit"
        elif rank < len(sorted_layers) - n_bf16:
            configs[layer_idx] = "8bit"
        else:
            configs[layer_idx] = "bf16"

    return configs, sorted_layers


def main():
    parser = argparse.ArgumentParser(
        description="Compare BF16 vs Uniform GPTQ vs QRP+GPTQ mixed precision"
    )
    parser.add_argument("--model-name", type=str,
                        default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--samples", type=int, default=10,
                        help="Samples for GSM8K eval and target prob")
    parser.add_argument("--pct-bf16", type=int, default=30,
                        help="Percentage of top layers to keep at bf16")
    parser.add_argument("--pct-8bit", type=int, default=0,
                        help="Percentage of layers for 8-bit (middle)")
    parser.add_argument("--wikitext-tokens", type=int, default=4096,
                        help="Max tokens for WikiText-2 PPL eval")
    parser.add_argument("--group-size", type=int, default=64,
                        help="GPTQ group size (64 works for small models)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip if aggregated_scores.json doesn't exist "
                             "(use uniform quantization only)")
    args = parser.parse_args()

    model_safe = args.model_name.replace("/", "_")
    base_dir = os.path.join(args.output_folder, model_safe)
    out_dir = os.path.join(base_dir, "gptq_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    # Check for QRP scores
    aggregated_file = os.path.join(base_dir, "aggregated_scores.json")
    has_qrp_scores = os.path.exists(aggregated_file) and not args.skip_analysis

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  GPTQ MIXED-PRECISION BENCHMARK")
    print(f"  Model: {args.model_name}")
    print(f"{'='*65}\n")

    print("Loading model and preparing calibration data...")
    quantizer = GPTQMixedQuantizer(args.model_name)
    num_layers = quantizer.num_layers
    calibration_data = quantizer.prepare_calibration_data(
        dataset_name="wikitext", n_samples=32, seq_len=256
    )
    print(f"Model has {num_layers} layers")
    print(f"Prepared {len(calibration_data)} calibration chunks\n")

    # ── Build configs to evaluate ───────────────────────────────────────
    configs_to_eval = {}

    # Config 1: BF16 baseline — no quantization
    configs_to_eval["BF16 (baseline)"] = {}

    # Config 2: Uniform GPTQ 4-bit — all layers quantized
    configs_to_eval["Uniform GPTQ-4bit"] = {
        i: "4bit" for i in range(num_layers)
    }

    # Config 3: QRP + GPTQ mixed — only if QRP scores exist
    if has_qrp_scores:
        qrp_config, sorted_layers = load_qrp_precision_map(
            aggregated_file, num_layers,
            pct_bf16=args.pct_bf16, pct_8bit=args.pct_8bit
        )
        n4 = sum(1 for v in qrp_config.values() if v == "4bit")
        n8 = sum(1 for v in qrp_config.values() if v == "8bit")
        n_bf = sum(1 for v in qrp_config.values() if v == "bf16")
        label = f"QRP+GPTQ ({n4}×4bit, {n8}×8bit, {n_bf}×bf16)"
        configs_to_eval[label] = qrp_config

        print(f"QRP precision map loaded: {n4}×4-bit, {n8}×8-bit, {n_bf}×bf16")
        print(f"Layer scores: {sorted_layers[0][1]:.4f} — {sorted_layers[-1][1]:.4f}")
    else:
        print("No QRP scores found — skipping QRP+GPTQ mixed config.")
        print(f"(Expected file: {aggregated_file})")
        print("Run the analysis pipeline first, or use --skip-analysis\n")

    # ── Evaluate each config ────────────────────────────────────────────
    results = []

    for config_name, layer_config in configs_to_eval.items():
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {config_name}")
        print(f"{'─'*60}")

        # Apply quantization
        if layer_config:
            quantizer.quantize_with_precision_map(
                layer_config,
                calibration_data=calibration_data,
                group_size=args.group_size
            )
        else:
            quantizer.restore()

        # Evaluate WikiText-2 PPL
        print("\n  [1/3] WikiText-2 Perplexity...")
        ppl = evaluate_wikitext2_ppl(
            quantizer.model, quantizer.tokenizer,
            max_tokens=args.wikitext_tokens
        )
        print(f"  WikiText-2 PPL: {ppl:.2f}")

        # Evaluate GSM8K exact-match
        print(f"\n  [2/3] GSM8K Exact-Match ({args.samples} samples)...")
        gsm_result = evaluate_gsm8k_exact_match(
            quantizer.model, quantizer.tokenizer,
            n_samples=args.samples
        )
        print(f"  GSM8K Accuracy: {gsm_result['accuracy']:.2%} "
              f"({gsm_result['correct']}/{gsm_result['total']})")

        # Evaluate target prob (existing metric)
        print(f"\n  [3/3] Target Probability ({args.samples} samples)...")
        target_prob = evaluate_target_prob(
            quantizer.model, quantizer.tokenizer,
            dataset_name="gsm8k", n_samples=args.samples
        )
        print(f"  Target Prob: {target_prob:.4f}")

        # Estimate size
        from qrp.quantize.run_multi_dataset_benchmark import estimate_size
        est_size = estimate_size(
            quantizer.model, layer_config, num_layers
        )
        est_size_mb = est_size / 1e6

        results.append({
            "config": config_name,
            "wikitext2_ppl": ppl,
            "gsm8k_accuracy": gsm_result["accuracy"],
            "gsm8k_correct": gsm_result["correct"],
            "gsm8k_total": gsm_result["total"],
            "target_prob": target_prob,
            "estimated_size_mb": est_size_mb,
            "gsm8k_results": gsm_result["results"],
        })

        # Restore for next config
        quantizer.restore()

    # ── Print comparison table ──────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  RESULTS COMPARISON")
    print(f"{'='*80}")
    header = (f"  {'Config':<35} {'Size(MB)':>9} {'WText PPL':>10} "
              f"{'GSM8K Acc':>10} {'Tgt Prob':>9}")
    print(header)
    print("  " + "─" * 75)

    for r in results:
        print(f"  {r['config']:<35} {r['estimated_size_mb']:>8.1f} "
              f"{r['wikitext2_ppl']:>10.2f} "
              f"{r['gsm8k_accuracy']:>9.2%} "
              f"{r['target_prob']:>9.4f}")

    # ── Compute relative metrics ────────────────────────────────────────
    if len(results) >= 2:
        baseline = results[0]
        print(f"\n  Relative to BF16 baseline:")
        for r in results[1:]:
            ppl_change = r['wikitext2_ppl'] / baseline['wikitext2_ppl'] - 1
            size_reduction = (1 - r['estimated_size_mb'] / baseline['estimated_size_mb']) * 100
            print(f"  {r['config']:<35} "
                  f"PPL: {'+' if ppl_change >= 0 else ''}{ppl_change:.1%}  "
                  f"Size: -{size_reduction:.1f}%")

    # ── Save results ────────────────────────────────────────────────────
    save_results = []
    for r in results:
        save_r = {k: v for k, v in r.items() if k != "gsm8k_results"}
        save_results.append(save_r)

    json_path = os.path.join(out_dir, "gptq_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "samples": args.samples,
            "wikitext_tokens": args.wikitext_tokens,
            "group_size": args.group_size,
            "pct_bf16": args.pct_bf16,
            "results": save_results,
        }, f, indent=2)

    # CSV
    csv_path = os.path.join(out_dir, "gptq_benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "estimated_size_mb", "wikitext2_ppl",
            "gsm8k_accuracy", "target_prob"
        ])
        writer.writeheader()
        for r in save_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # LaTeX
    tex_path = os.path.join(out_dir, "gptq_benchmark_results.tex")
    _write_latex(save_results, args.model_name, tex_path)

    print(f"\n  Results saved to: {out_dir}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  TeX:  {tex_path}")


def _write_latex(results, model_name, path):
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        f"  \\caption{{GPTQ Benchmark: {model_name.split('/')[-1]}}}",
        r"  \label{tab:gptq-benchmark}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    \textbf{Config} & \textbf{Size (MB)} & \textbf{WikiText-2 PPL $\downarrow$} "
        r"& \textbf{GSM8K Acc $\uparrow$} & \textbf{Target Prob $\uparrow$} \\",
        r"    \midrule",
    ]
    for r in results:
        config = r["config"].replace("×", r"$\times$").replace("%", r"\%")
        lines.append(
            f"    {config} & {r['estimated_size_mb']:.1f} & "
            f"{r['wikitext2_ppl']:.2f} & {r['gsm8k_accuracy']:.2%} & "
            f"{r['target_prob']:.4f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
