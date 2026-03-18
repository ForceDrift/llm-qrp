import os
import argparse
import json
import torch
import math
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def main():
    parser = argparse.ArgumentParser(description="Empirical Threshold Testing for Quantization on lowest-scoring layers")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Model name evaluated")
    parser.add_argument("--output-folder", type=str, required=True, help="Base folder where aggregated scores are saved")
    parser.add_argument("--samples", type=int, default=10, help="Number of questions to evaluate for probability")
    
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
    
    thresholds = [10, 20, 30, 40, 50]
    
    print("\nLoading dataset and running baseline quantizer...")
    quantizer = TargetedQuantizer(args.model_name)
    ds = load_dataset("gsm8k", "main", split="test")
    eval_dataset = list(ds)[:args.samples]
    
    print("\nEvaluating Baseline (BF16)")
    baseline_acc = evaluate_gsm8k(quantizer.model, quantizer.tokenizer, eval_dataset)
    print(f"Baseline Target Prob: {baseline_acc:.2%}")
    
    results = {
        "baseline_prob": baseline_acc,
        "performance_8bit": [],
        "performance_4bit": [],
        "performance_mixed": []
    }
    
    for pct in thresholds:
        layer_count = max(1, int(num_layers * (pct / 100.0)))
        target_layers = sorted_layer_indices[:layer_count]
        
        print(f"\n--- Testing Threshold: Bottom {pct}% ({layer_count} layers) ---")
        print(f"Layers to quantize: {target_layers}")
        
        # 8-bit
        quantizer.quantize_layers({idx: "8bit" for idx in target_layers})
        acc_8 = evaluate_gsm8k(quantizer.model, quantizer.tokenizer, eval_dataset)
        results["performance_8bit"].append((pct, acc_8))
        print(f"8-bit Prob: {acc_8:.2%}")
        
        # 4-bit
        quantizer.quantize_layers({idx: "4bit" for idx in target_layers})
        acc_4 = evaluate_gsm8k(quantizer.model, quantizer.tokenizer, eval_dataset)
        results["performance_4bit"].append((pct, acc_4))
        print(f"4-bit Prob: {acc_4:.2%}")
        
        # Mixed (bottom half 4-bit, upper half 8-bit)
        mixed_configs = {}
        half_idx = len(target_layers) // 2
        for i, idx in enumerate(target_layers):
            mixed_configs[idx] = "4bit" if i < half_idx else "8bit"
            
        quantizer.quantize_layers(mixed_configs)
        acc_mixed = evaluate_gsm8k(quantizer.model, quantizer.tokenizer, eval_dataset)
        results["performance_mixed"].append((pct, acc_mixed))
        print(f"Mixed Prob: {acc_mixed:.2%}")
        
    quantize_out_dir = os.path.join(base_dir, "quantize")
    os.makedirs(quantize_out_dir, exist_ok=True)
    json_path = os.path.join(quantize_out_dir, "quantize_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nGenerating Empirical Quantization Graph...")
    plt.figure(figsize=(10, 6))
    
    x_bits = [0] + [x[0] for x in results["performance_8bit"]]
    y_8 = [baseline_acc] + [x[1] for x in results["performance_8bit"]]
    y_4 = [baseline_acc] + [x[1] for x in results["performance_4bit"]]
    y_mixed = [baseline_acc] + [x[1] for x in results["performance_mixed"]]
    
    plt.plot(x_bits, y_8, marker='o', linestyle='-', color='blue', label='8-bit Quantization')
    plt.plot(x_bits, y_4, marker='x', linestyle='--', color='red', label='4-bit Quantization')
    plt.plot(x_bits, y_mixed, marker='s', linestyle='-.', color='green', label='Mixed (4-bit lowest, 8-bit med)')
    plt.axhline(y=baseline_acc, color='gray', linestyle=':', label='Baseline BF16')
    
    plt.title('Performance Drop from Quantizing Bottom Thinking Layers')
    plt.xlabel('Percentage of Lowest Scoring Layers Quantized (%)')
    plt.ylabel('Target Prob (exp(-loss))')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks([0] + thresholds)
    
    img_path = os.path.join(quantize_out_dir, "quantization_empirical_drops.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved graph to {img_path}")
    print(f"Saved metrics to {json_path}")
    
if __name__ == "__main__":
    main()
