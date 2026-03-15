import os
import torch
import numpy as np
from collections import defaultdict
from .sled import SLED_Decoded
from .entropy_by_layer import EntropyByLayer


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