"""
ablation_impact_analysis.py — Analytical measure of how ablation affects "thinking".

This script:
  1. Loads a model and its precomputed importance scores.
  2. Selects "Thinking" (top-score) and "Redundant" (bottom-score) layers.
  3. For each selected layer:
     a. Measures SLED/Entropy on a prompt with the model INTACT.
     b. Measures SLED/Entropy with that layer ABLATED.
  4. Compares the resulting "Disagreement Map" to show how thinking collapses.
"""

import os
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from qrp.analysis.sled import SLED_Decoded
from qrp.analysis.entropy_by_layer import EntropyByLayer
from qrp.analysis.ablation_controller import AblationController
from qrp.analysis.aggregate_scores import getSortedLayers

class AblationImpactAnalyzer:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.ablation_ctrl = AblationController(model_name)
        # We share the model across all tools
        self.sled_model = SLED_Decoded(model_name, self.device)
        self.sled_model.model = self.ablation_ctrl.model
        
        self.entropy_model = EntropyByLayer(model_name, device=self.device)
        self.entropy_model.model = self.ablation_ctrl.model

    def get_baseline(self, prompt):
        self.ablation_ctrl.restore_layers()
        return self._run_metrics(prompt)

    def get_ablated_impact(self, prompt, layer_idx):
        self.ablation_ctrl.ablate_layers([layer_idx])
        metrics = self._run_metrics(prompt)
        self.ablation_ctrl.restore_layers()
        return metrics

    def _run_metrics(self, prompt):
        # Measure SLED Disagreement
        layers = list(range(self.ablation_ctrl.layer_count))
        disagreement = self.sled_model.layer_disagreement(prompt, evolution_scale=5, candidate_premature_layers=layers)
        
        # Average across tokens
        layer_scores = {}
        for layer_idx in layers:
            token_scores = [d[layer_idx].mean().item() for d in disagreement if layer_idx in d]
            if token_scores:
                layer_scores[f"layer_{layer_idx}"] = float(np.mean(token_scores))
        
        return layer_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--scores-file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    # Load scores
    with open(args.scores_file, "r") as f:
        score_data = json.load(f)
    sorted_layers = getSortedLayers(score_data["layerAvgScores"])

    # Pick top and bottom layer to test
    target_layers = {
        "thinking": sorted_layers[-1][0], # Most important
        "redundant": sorted_layers[0][0]  # Least important
    }

    analyzer = AblationImpactAnalyzer(args.model_name)
    ds = load_dataset("gsm8k", "main", split="test")
    prompts = [x["question"] for x in ds][:args.limit]

    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Analyzing impact")):
        baseline = analyzer.get_baseline(prompt)
        thinking_impact = analyzer.get_ablated_impact(prompt, target_layers["thinking"])
        redundant_impact = analyzer.get_ablated_impact(prompt, target_layers["redundant"])

        results.append({
            "id": i,
            "prompt": prompt[:100] + "...",
            "baseline": baseline,
            "ablate_thinking_layer": target_layers["thinking"],
            "thinking_impact": thinking_impact,
            "ablate_redundant_layer": target_layers["redundant"],
            "redundant_impact": redundant_impact
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nImpact analysis complete: {args.output_file}")

if __name__ == "__main__":
    main()
