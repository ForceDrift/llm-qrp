import os
import argparse
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

from qrp.analysis.ablation_controller import AblationController


def extract_answer(text):
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = re.sub(r"[^\d\.-]", "", ans)
        return ans
    
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        return numbers[-1]
    return ""

def evaluate_gsm8k(controller, dataset, max_new_tokens=256):
    correct = 0
    total = len(dataset)
    
    model = controller.model
    tokenizer = controller.tokenizer
    
    model.eval()
    
    for item in tqdm(dataset, desc="Evaluating GSM8K", leave=False):
        question = item["question"]
        expected_ans = extract_answer(item["answer"])
        
        prompt = f"Question: {question}\nAnswer: Let's think step by step\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_ans = extract_answer(generated_text)
        
        if pred_ans == expected_ans and expected_ans != "":
            correct += 1
            
    return correct / total if total > 0 else 0


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
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    
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
        print(f"Accuracy: {acc:.2%}")
        
    print("\n--- Progressively Ablating Top 20% Layers ---")
    current_ablation_list = []
    
    for i, layer_idx in enumerate(top_20_percent):
        current_ablation_list.append(layer_idx)
        print(f"Ablating {len(current_ablation_list)} layers: {current_ablation_list}")
        
        controller.ablateLayer(current_ablation_list)
        acc = evaluate_gsm8k(controller, eval_dataset)
        results["top_20_ablation"].append((len(current_ablation_list), acc))
        
        controller.restoreLayers()
        print(f"Accuracy: {acc:.2%}")
        
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
    
    plt.title('GSM8K Accuracy vs Number of Layers Ablated')
    plt.xlabel('Number of Layers Ablated')
    plt.ylabel('Exact Match Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, twenty_percent_count + 1))
    
    img_path = os.path.join(ablation_out_dir, "ablation_performance.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Saved graph to {img_path}")
    print(f"Saved metrics to {json_path}")
    
    
if __name__ == "__main__":
    main()
