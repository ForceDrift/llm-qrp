import os
import argparse
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset
from tqdm import tqdm

from qrp.quantize.quantizer import TargetedQuantizer


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

    avg_loss = total_loss / valid_samples if valid_samples > 0 else float('inf')
    target_prob_score = math.exp(-avg_loss) if avg_loss != float('inf') else 0.0
    return target_prob_score


def estimate_layer_params(model, layer_idx):
    """Count the total weight parameters in a transformer layer's projections."""
    layer = model.model.layers[layer_idx]
    total_params = 0

    attn_projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_projs = ["gate_proj", "up_proj", "down_proj"]

    for name in attn_projs:
        proj = getattr(layer.self_attn, name, None)
        if proj is not None:
            total_params += proj.weight.numel()

    for name in mlp_projs:
        proj = getattr(layer.mlp, name, None)
        if proj is not None:
            total_params += proj.weight.numel()

    return total_params


def estimate_model_size(model, layer_configs, num_layers):
    """Estimate model size in bytes given a per-layer precision config."""
    total_bytes = 0.0
    bytes_per_param = {"bf16": 2.0, "8bit": 1.0, "4bit": 0.5}

    for idx in range(num_layers):
        params = estimate_layer_params(model, idx)
        precision = layer_configs.get(idx, "bf16")
        total_bytes += params * bytes_per_param[precision]

    return total_bytes


def generate_candidate_configs(sorted_layers, num_layers):
    """
    For each pair:
      - Bottom pct_4bit% of layers (by score) -> 4-bit
      - Next (pct_8bit - pct_4bit)% of layers -> 8-bit
      - Remaining layers -> BF16
    """
    percentiles = list(range(0, 101, 10))  
    candidates = []

    for p4 in percentiles:
        for p8 in percentiles:
            if p4 > p8:
                continue

            n4 = int(num_layers * (p4 / 100.0))
            n8 = int(num_layers * (p8 / 100.0))

            config = {}
            for rank, (layer_idx, score) in enumerate(sorted_layers):
                if rank < n4:
                    config[layer_idx] = "4bit"
                elif rank < n8:
                    config[layer_idx] = "8bit"

            candidates.append({
                "pct_4bit": p4,
                "pct_8bit": p8,
                "n_4bit": n4,
                "n_8bit_only": n8 - n4,
                "n_bf16": num_layers - n8,
                "layer_configs": config
            })

    return candidates


def generate_topology_graph(model_name, num_layers, best_config, layer_scores_sorted,
                            baseline_acc, best_acc, best_efficiency,
                            baseline_size, best_size, output_path):
    """Generate a model topology graph showing per-layer precision assignments."""

    # Build per-layer info
    layer_info = {}
    score_dict = {idx: score for idx, score in layer_scores_sorted}
    for idx in range(num_layers):
        precision = best_config.get(idx, "bf16")
        score = score_dict.get(idx, 0.0)
        layer_info[idx] = {"precision": precision, "score": score}

    color_map = {"bf16": "#4CAF50", "8bit": "#FFC107", "4bit": "#F44336"}
    label_map = {"bf16": "BF16 (Full)", "8bit": "INT8", "4bit": "FP4"}

    fig_height = max(8, 2 + num_layers * 0.45)
    fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))

    box_width = 5.0
    box_height = 0.6
    gap = 0.15
    x_center = 5.0
    x_left = x_center - box_width / 2

    # Draw from top to bottom: embedding -> layers -> lm_head
    total_stack_height = (num_layers + 2) * (box_height + gap)
    y_top = total_stack_height

    # --- Embedding layer ---
    y = y_top
    rect = plt.Rectangle((x_left, y), box_width, box_height,
                          facecolor="#9E9E9E", edgecolor="#333", linewidth=1.2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x_center, y + box_height / 2, "Token Embedding",
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # --- Transformer layers ---
    for idx in range(num_layers):
        y = y_top - (idx + 1) * (box_height + gap)
        info = layer_info[idx]
        color = color_map[info["precision"]]
        label = label_map[info["precision"]]

        rect = plt.Rectangle((x_left, y), box_width, box_height,
                              facecolor=color, edgecolor="#333", linewidth=1.0, alpha=0.85)
        ax.add_patch(rect)

        left_text = f"Layer {idx}"
        right_text = f"{label}  (score: {info['score']:.3f})"
        ax.text(x_left + 0.2, y + box_height / 2, left_text,
                ha='left', va='center', fontsize=8, fontweight='bold')
        ax.text(x_left + box_width - 0.2, y + box_height / 2, right_text,
                ha='right', va='center', fontsize=7.5, color='#333')

    # --- LM Head ---
    y = y_top - (num_layers + 1) * (box_height + gap)
    rect = plt.Rectangle((x_left, y), box_width, box_height,
                          facecolor="#9E9E9E", edgecolor="#333", linewidth=1.2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x_center, y + box_height / 2, "LM Head",
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # --- Arrows between layers ---
    for idx in range(num_layers + 1):
        y_arrow_top = y_top - idx * (box_height + gap)
        y_arrow_bottom = y_top - (idx + 1) * (box_height + gap) + box_height
        ax.annotate('', xy=(x_center, y_arrow_bottom),
                    xytext=(x_center, y_arrow_top),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.0))

    # --- Legend ---
    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="BF16 (Full Precision)"),
        mpatches.Patch(color="#FFC107", label="INT8 (8-bit)"),
        mpatches.Patch(color="#F44336", label="FP4 (4-bit)"),
        mpatches.Patch(color="#9E9E9E", label="Fixed (not quantized)"),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8,
              framealpha=0.9, edgecolor='#ccc')

    # --- Summary stats ---
    size_reduction_pct = (1 - best_size / baseline_size) * 100 if baseline_size > 0 else 0
    acc_drop_pct = (1 - best_acc / baseline_acc) * 100 if baseline_acc > 0 else 0

    n_4bit = sum(1 for v in best_config.values() if v == "4bit")
    n_8bit = sum(1 for v in best_config.values() if v == "8bit")
    n_bf16 = num_layers - n_4bit - n_8bit

    summary_text = (
        f"Model: {model_name}\n"
        f"Layers: {n_4bit} × FP4  |  {n_8bit} × INT8  |  {n_bf16} × BF16\n"
        f"Size Reduction: {size_reduction_pct:.1f}%  |  "
        f"Accuracy Retained: {(100 - acc_drop_pct):.1f}%  |  "
        f"Efficiency: {best_efficiency:.2f}"
    )

    ax.text(x_center, y - 0.6, summary_text,
            ha='center', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5',
                      edgecolor='#ccc', alpha=0.95),
            family='monospace')

    ax.set_xlim(x_left - 1, x_left + box_width + 1)
    ax.set_ylim(y - 1.8, y_top + box_height + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Optimal Mixed-Precision Model Topology\n{model_name}",
                 fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved topology graph to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find the optimal mixed-precision quantization config "
                    "by sweeping threshold pairs and maximizing compression/accuracy efficiency"
    )
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M",
                        help="Model name to evaluate")
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Base folder where aggregated scores are saved")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of GSM8K questions to evaluate for probability")
    parser.add_argument("--min-accuracy-floor", type=float, default=0.5,
                        help="Minimum fraction of baseline accuracy to accept (0.0-1.0)")

    args = parser.parse_args()

    model_name_safe = args.model_name.replace("/", "_")
    base_dir = os.path.join(args.output_folder, model_name_safe)
    aggregated_file = os.path.join(base_dir, "aggregated_scores.json")

    if not os.path.exists(aggregated_file):
        raise FileNotFoundError(
            f"Could not find {aggregated_file}. Please run run_analysis.py first."
        )

    with open(aggregated_file, "r") as f:
        scores = json.load(f)

    # Sort layers by score ascending (lowest = least important for reasoning)
    sorted_layers = sorted(
        [(int(key.split("_")[1]), val) for key, val in scores.items()],
        key=lambda x: x[1]
    )
    num_layers = len(sorted_layers)

    print(f"\nLoaded {num_layers} layer scores from {aggregated_file}")
    print(f"Layer score range: {sorted_layers[0][1]:.4f} — {sorted_layers[-1][1]:.4f}")

    # --- Load model and dataset ---
    print("\nLoading model and dataset...")
    quantizer = TargetedQuantizer(args.model_name)
    ds = load_dataset("gsm8k", "main", split="test")
    eval_dataset = list(ds)[:args.samples]

    # --- Baseline evaluation ---
    print("\nEvaluating Baseline (BF16, no quantization)...")
    baseline_acc = evaluate_gsm8k(quantizer.model, quantizer.tokenizer, eval_dataset)
    print(f"Baseline Target Prob: {baseline_acc:.4f}")

    baseline_size = estimate_model_size(quantizer.model, {}, num_layers)
    print(f"Baseline estimated layer size: {baseline_size / 1e6:.2f} MB")

    # --- Generate and evaluate candidates ---
    candidates = generate_candidate_configs(sorted_layers, num_layers)
    print(f"\nGenerated {len(candidates)} candidate configs. Evaluating each...\n")

    results = []
    accuracy_floor = baseline_acc * args.min_accuracy_floor

    for i, candidate in enumerate(tqdm(candidates, desc="Sweeping configs")):
        config = candidate["layer_configs"]

        # Skip the trivial all-BF16 config (no quantization)
        if not config:
            est_size = baseline_size
            results.append({
                **candidate,
                "accuracy": baseline_acc,
                "estimated_size_bytes": est_size,
                "size_reduction_pct": 0.0,
                "accuracy_drop_pct": 0.0,
                "efficiency": 0.0,  # No compression = 0 efficiency
            })
            continue

        # Convert keys to int for quantizer
        int_config = {int(k): v for k, v in config.items()}

        quantizer.quantize_layers(int_config)
        acc = evaluate_gsm8k(quantizer.model, quantizer.tokenizer, eval_dataset)

        est_size = estimate_model_size(quantizer.model, int_config, num_layers)
        size_reduction = (1 - est_size / baseline_size) * 100
        acc_drop = (1 - acc / baseline_acc) * 100 if baseline_acc > 0 else 0

        # Efficiency = size_reduction / accuracy_drop (higher = better)
        # Guard against zero or negative accuracy drop
        if acc_drop > 0.01:
            efficiency = size_reduction / acc_drop
        elif acc_drop <= 0.01 and size_reduction > 0:
            # Accuracy barely dropped but we got compression — excellent
            efficiency = size_reduction * 100
        else:
            efficiency = 0.0

        result = {
            **candidate,
            "accuracy": acc,
            "estimated_size_bytes": est_size,
            "size_reduction_pct": size_reduction,
            "accuracy_drop_pct": acc_drop,
            "efficiency": efficiency,
        }
        results.append(result)

        label = (f"[{candidate['pct_4bit']}% 4-bit, {candidate['pct_8bit']}% 8-bit] "
                 f"Acc: {acc:.4f} | Size↓: {size_reduction:.1f}% | Eff: {efficiency:.2f}")
        tqdm.write(label)

    # --- Select the best config ---
    # Filter to configs that meet the accuracy floor
    valid_results = [r for r in results if r["accuracy"] >= accuracy_floor]

    if not valid_results:
        print(f"\nWARNING: No configs met the accuracy floor of {accuracy_floor:.4f}.")
        print("Falling back to all results...")
        valid_results = results

    # Among valid configs that actually quantize something, pick highest efficiency
    quantized_results = [r for r in valid_results if r["size_reduction_pct"] > 0]

    if quantized_results:
        best = max(quantized_results, key=lambda x: x["efficiency"])
    else:
        best = max(valid_results, key=lambda x: x["efficiency"])

    print(f"\n{'='*60}")
    print(f"OPTIMAL MIXED-PRECISION CONFIG FOUND")
    print(f"{'='*60}")
    print(f"  4-bit layers: {best['n_4bit']} ({best['pct_4bit']}%)")
    print(f"  8-bit layers: {best['n_8bit_only']} (next {best['pct_8bit'] - best['pct_4bit']}%)")
    print(f"  BF16 layers:  {best['n_bf16']}")
    print(f"  Accuracy:     {best['accuracy']:.4f} (baseline: {baseline_acc:.4f})")
    print(f"  Size reduction: {best['size_reduction_pct']:.1f}%")
    print(f"  Accuracy drop:  {best['accuracy_drop_pct']:.1f}%")
    print(f"  Efficiency:     {best['efficiency']:.2f}")
    print(f"  Layer assignments: {best['layer_configs']}")

    # --- Save results ---
    out_dir = os.path.join(base_dir, "quantize")
    os.makedirs(out_dir, exist_ok=True)

    # Clean up non-serializable types
    serializable_results = []
    for r in results:
        clean = {k: v for k, v in r.items()}
        # Convert any numpy/torch types
        for k, v in clean.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
        serializable_results.append(clean)

    best_clean = {k: v for k, v in best.items()}
    for k, v in best_clean.items():
        if isinstance(v, (np.floating, np.integer)):
            best_clean[k] = float(v)

    output_data = {
        "model_name": args.model_name,
        "baseline_accuracy": baseline_acc,
        "baseline_size_bytes": baseline_size,
        "accuracy_floor": accuracy_floor,
        "optimal_config": best_clean,
        "all_candidates": serializable_results,
    }

    json_path = os.path.join(out_dir, "optimal_mixed_precision.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved optimal config to {json_path}")

    # --- Generate topology graph ---
    topology_path = os.path.join(out_dir, "model_topology.png")
    generate_topology_graph(
        model_name=args.model_name,
        num_layers=num_layers,
        best_config=best["layer_configs"],
        layer_scores_sorted=sorted_layers,
        baseline_acc=baseline_acc,
        best_acc=best["accuracy"],
        best_efficiency=best["efficiency"],
        baseline_size=baseline_size,
        best_size=best["estimated_size_bytes"],
        output_path=topology_path
    )

    # --- Generate efficiency frontier plot ---
    frontier_path = os.path.join(out_dir, "efficiency_frontier.png")
    _generate_efficiency_plot(results, best, baseline_acc, frontier_path)

    print(f"\nDone! All outputs saved to {out_dir}")


def _generate_efficiency_plot(results, best, baseline_acc, output_path):
    """Plot all evaluated configs on an accuracy vs. size-reduction scatterplot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sizes = [r["size_reduction_pct"] for r in results]
    accs = [r["accuracy"] for r in results]
    effs = [r["efficiency"] for r in results]

    scatter = ax.scatter(sizes, accs, c=effs, cmap='RdYlGn', s=50, alpha=0.7,
                         edgecolors='#333', linewidths=0.5)
    plt.colorbar(scatter, label='Efficiency Score', ax=ax)

    # Highlight the best config
    ax.scatter([best["size_reduction_pct"]], [best["accuracy"]],
               color='none', edgecolors='blue', s=200, linewidths=2.5,
               label=f'Optimal ({best["pct_4bit"]}% 4-bit, {best["pct_8bit"]}% 8-bit)',
               zorder=5)

    ax.axhline(y=baseline_acc, color='gray', linestyle=':', alpha=0.5, label='Baseline BF16')

    ax.set_xlabel('Size Reduction (%)', fontsize=11)
    ax.set_ylabel('Target Probability (Accuracy)', fontsize=11)
    ax.set_title('Mixed-Precision Efficiency Frontier', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved efficiency frontier plot to {output_path}")


if __name__ == "__main__":
    main()
