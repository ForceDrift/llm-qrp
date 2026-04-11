import argparse
import json
import math
import os
import re

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from tqdm import tqdm

from qrp.analysis.ablation_controller import AblationController


def evaluate_gsm8k(controller, dataset, max_new_tokens=256):
    total_loss = 0.0
    valid_samples = 0
    model = controller.model
    tokenizer = controller.tokenizer
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate performance of ablating top/bottom 20% thinking layers")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M", help="Model to evaluate")
    parser.add_argument("--output-folder", type=str, required=True, help="Base folder where results are saved")
    parser.add_argument("--samples", type=int, default=10, help="Number of questions to evaluate at each step for speed")
    args = parser.parse_args()

    model_name_safe = args.model_name.replace("/", "_")
    base_dir = os.path.join(args.output_folder, model_name_safe)
    aggregated_file = os.path.join(base_dir, "aggregated_scores.json")

    if not os.path.exists(aggregated_file):
        raise FileNotFoundError(f"Could not find {aggregated_file}. Please run run_analysis.py first.")

    with open(aggregated_file, "r") as f:
        scores = json.load(f)

    sorted_layers = sorted(scores.items(), key=lambda item: item[1])
    sorted_layer_indices = [int(key.split("_")[1]) for key, val in sorted_layers]
    num_layers = len(sorted_layer_indices)
    twenty_percent_count = max(1, int(num_layers * 0.2))

    bottom_20_percent = sorted_layer_indices[:twenty_percent_count]
    top_20_percent = sorted_layer_indices[-twenty_percent_count:]

    print(f"Total layers: {num_layers}")
    print(f"Number of layers to ablate in 20% split: {twenty_percent_count}")
    print(f"Bottom 20% (Lowest Thinking Scores): {bottom_20_percent}")
    print(f"Top 20% (Highest Thinking Scores): {top_20_percent}")

    print("\nLoading dataset and model...")
    controller = AblationController(args.model_name)
    ds = load_dataset("gsm8k", "main", split="test")
    eval_dataset = list(ds)[:args.samples]

    print("\nEvaluating Baseline (No Ablation)")
    baseline_acc = evaluate_gsm8k(controller, eval_dataset)
    print(f"Baseline Target Prob: {baseline_acc:.2%}")

    results = {
        "baseline_accuracy": baseline_acc,
        "bottom_20_ablation": [],
        "top_20_ablation": []
    }

    print("\n--- Progressively Ablating Bottom 20% Layers ---")
    current_ablation_list = []
    for i, layer_idx in enumerate(bottom_20_percent):
        current_ablation_list.append(layer_idx)
        print(f"Ablating {len(current_ablation_list)} layers: {current_ablation_list}")
        controller.ablateLayer(current_ablation_list)
        acc = evaluate_gsm8k(controller, eval_dataset)
        results["bottom_20_ablation"].append((len(current_ablation_list), acc))
        controller.restoreLayers()
        print(f"Target Prob: {acc:.2%}")

    print("\n--- Progressively Ablating Top 20% Layers ---")
    current_ablation_list = []
    for i, layer_idx in enumerate(top_20_percent):
        current_ablation_list.append(layer_idx)
        print(f"Ablating {len(current_ablation_list)} layers: {current_ablation_list}")
        controller.ablateLayer(current_ablation_list)
        acc = evaluate_gsm8k(controller, eval_dataset)
        results["top_20_ablation"].append((len(current_ablation_list), acc))
        controller.restoreLayers()
        print(f"Target Prob: {acc:.2%}")

    ablation_out_dir = os.path.join(base_dir, "ablation")
    os.makedirs(ablation_out_dir, exist_ok=True)
    json_path = os.path.join(ablation_out_dir, "ablation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nGenerating Performance Graph...")
    plt.figure(figsize=(10, 6))

    x_bottom = [0] + [x[0] for x in results["bottom_20_ablation"]]
    y_bottom = [baseline_acc] + [x[1] for x in results["bottom_20_ablation"]]
    x_top = [0] + [x[0] for x in results["top_20_ablation"]]
    y_top = [baseline_acc] + [x[1] for x in results["top_20_ablation"]]

    plt.plot(x_bottom, y_bottom, marker='o', linestyle='-', color='blue', label='Removing Bottom 20% (Lowest Thinking)')
    plt.plot(x_top, y_top, marker='x', linestyle='--', color='red', label='Removing Top 20% (Highest Thinking)')
    plt.axhline(y=baseline_acc, color='gray', linestyle=':', label='Baseline Performance')
    plt.title('GSM8K Target Prob vs Number of Layers Ablated')
    plt.xlabel('Number of Layers Ablated')
    plt.ylabel('Target Prob (exp(-loss))')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, twenty_percent_count + 1))

    img_path = os.path.join(ablation_out_dir, "ablation_performance.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Saved graph to {img_path}")
    print(f"Saved metrics to {json_path}")


if __name__ == "__main__":
    main()

