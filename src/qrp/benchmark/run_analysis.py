import os
import torch
import numpy as np
from collections import defaultdict
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

from qrp.analysis.sled import SLED_Decoded
from qrp.analysis.entropy_by_layer import EntropyByLayer

class SLEDEntropyAnalyzer:

    def __init__(self, model_name="HuggingFaceTB/SmolLM2-360M", dataset="gsm8k"):
        self.model_name = model_name
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sled_model = SLED_Decoded(model_name, self.device)
        self.entropy_model = EntropyByLayer(model_name, device=self.device)

    def _minMaxScale(self, data_dict):
        values = list(data_dict.values())
        if not values:
            return data_dict

        minVal = np.min(values)
        maxVal = np.max(values)
        if maxVal == minVal:
            return {k: 0.0 for k in data_dict}

        return {k: (v - minVal) / (maxVal - minVal) for k, v in data_dict.items()}

    def run(self, prompt):
        layers_to_test = list(range(self.sled_model.layers))

        disagreement_results = self.sled_model.layer_disagreement(
            prompt,
            evolution_scale=5,
            candidate_premature_layers=layers_to_test
        )

        layer_disagreement = defaultdict(list)
        for token_data in disagreement_results:
            for layer_idx, scores in token_data.items():
                layer_disagreement[layer_idx].append(scores.mean().item())

        layer_disagreement_mean = {
            f"layer_{k}": float(torch.tensor(v).mean())
            for k, v in layer_disagreement.items()
        }

        kl_prev = self.sled_model.kl_between_current_prev(prompt)
        kl_prev_dict = {
            f"layer_{i}": float(kl.item())
            for i, kl in enumerate(kl_prev, 1)
        }

        scaled_disagreement = self._minMaxScale(layer_disagreement_mean)
        scaled_kl = self._minMaxScale(kl_prev_dict)

        combined_average = {}
        for k in scaled_disagreement:
            if k in scaled_kl:
                combined_average[k] = (scaled_disagreement[k] + scaled_kl[k]) / 2

        entropy_results_raw = self.entropy_model.runStructuralAnalysis(prompt)
        entropy_dict = {f"layer_{res['layer']}": res['deltaH'] for res in entropy_results_raw}
        scaled_entropy = self._minMaxScale(entropy_dict)

        minmax_combined_all = {}
        for k in combined_average:
            if k in scaled_entropy:
                minmax_combined_all[k] = (combined_average[k] + scaled_entropy[k]) / 2

        return minmax_combined_all


def load_prompts(dataset_name):
    """Load prompts from supported datasets."""
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [x["question"] for x in ds]
    elif dataset_name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        return [x["question"] for x in ds]
    elif dataset_name.lower() == "truthfulqa":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        return [x["question"] for x in ds]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def aggregate_scores(results_data, output_file):
    """
    Aggregates layer score data across all processed prompts.
    """
    layer_sums = defaultdict(float)
    layer_counts = defaultdict(int)

    for item in results_data:
        analysis = item.get("analysis", {})
        for layer_key, score in analysis.items():
            layer_sums[layer_key] += score
            layer_counts[layer_key] += 1
            
    avg_scores = {}
    for layer_key, total in layer_sums.items():
        avg_scores[layer_key] = total / layer_counts[layer_key]
        
    with open(output_file, "w") as f:
        json.dump(avg_scores, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLED Entropy Analysis")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M", help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to evaluate (gsm8k, mmlu)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to analyze")
    parser.add_argument("--output-folder", type=str, required=True, help="Folder to save results")

    args = parser.parse_args()

    model_name_safe = args.model_name.replace("/", "_")
    output_dir = os.path.join(args.output_folder, model_name_safe, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    analyzer = SLEDEntropyAnalyzer(model_name=args.model_name, dataset=args.dataset)

    prompts = load_prompts(args.dataset)
    prompts = prompts[:args.samples]

    results = []

    for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts", unit="prompt")):
        analysis = analyzer.run(prompt)
        results.append({
            "id": idx,
            "prompt": prompt,
            "analysis": analysis
        })

    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Aggregate layer data into a single JSON at the root of the {model_name} folder
    root_dir = os.path.join(args.output_folder, model_name_safe)
    aggregated_file = os.path.join(root_dir, "aggregated_scores.json")
    aggregate_scores(results, aggregated_file)
    print(f"Aggregated scores saved to: {aggregated_file}")