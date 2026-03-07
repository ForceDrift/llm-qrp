# import os
# import json
# import torch
# from collections import defaultdict
# from sled import SLED_Decoded

# if __name__ == "__main__":
#     model_name = "EleutherAI/gpt-neo-1.3B"
#     dataset = "gsm8k"
#     prompt = "You discover a new color that no human has ever seen. How would you describe it to someone?"
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     test_model = SLED_Decoded(model_name, device)
#     layers_to_test = list(range(test_model.layers))

#     disagreement_results = test_model.layer_disagreement(
#         prompt, evolution_scale=5, candidate_premature_layers=layers_to_test
#     )
#     layer_scores = defaultdict(list)
#     for token_data in disagreement_results:
#         for layer_idx, scores in token_data.items():
#             layer_scores[layer_idx].append(scores.mean().item())
#     layer_scores_mean = {f"layer_{k}": float(torch.tensor(v).mean()) for k, v in layer_scores.items()}

#     kl_prev = test_model.kl_between_current_prev(prompt)
#     kl_prev_dict = {f"layer_{i}": float(kl.item()) for i, kl in enumerate(kl_prev, 1)}


#     token_ranks = test_model.token_ranking_evolution(prompt)
#     ranking_results = []
#     for r in token_ranks:
#         ranking_results.append({str(k): v.tolist() for k, v in r.items()})

#     sled_folder = f"results/{model_name}/{dataset}/sled_full"
#     os.makedirs(sled_folder, exist_ok=True)

#     sled_full_path = os.path.join(sled_folder, "sled_full.json")
#     with open(sled_full_path, "w") as f:
#         json.dump({
#             "model": model_name,
#             "dataset": dataset,
#             "layer_disagreement": layer_scores_mean,
#             "kl_between_layers": kl_prev_dict,
    
#             "token_ranking_evolution": ranking_results
#         }, f, indent=2)

#     print(f"Full SLED + stats results saved to {sled_full_path}")
import os
import json
import torch
import numpy as np
from collections import defaultdict
from sled import SLED_Decoded
from entropy_by_layer import EntropyByLayer  

def minMaxScale(data_dict):
    values = list(data_dict.values())
    if not values:
        return data_dict
    minVal = np.min(values)
    maxVal = np.max(values)
    if maxVal == minVal:
        return {k: 0.0 for k in data_dict}
    return {k: (v - minVal) / (maxVal - minVal) for k, v in data_dict.items()}

if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM2-360M"
    dataset = "gsm8k"
    prompt = "You discover a new color that no human has ever seen. How would you describe it to someone?"
    device = "cuda" if torch.cuda.is_available() else "cpu"


    test_model = SLED_Decoded(model_name, device)
    layers_to_test = list(range(test_model.layers))

    disagreement_results = test_model.layer_disagreement(
        prompt, evolution_scale=5, candidate_premature_layers=layers_to_test
    )
    
    layer_disagreement = defaultdict(list)
    for token_data in disagreement_results:
        for layer_idx, scores in token_data.items():
            layer_disagreement[layer_idx].append(scores.mean().item())
    layer_disagreement_mean = {f"layer_{k}": float(torch.tensor(v).mean()) 
                               for k, v in layer_disagreement.items()}

    kl_prev = test_model.kl_between_current_prev(prompt)
    kl_prev_dict = {f"layer_{i}": float(kl.item()) for i, kl in enumerate(kl_prev, 1)}

    scaled_disagreement = minMaxScale(layer_disagreement_mean)
    scaled_kl = minMaxScale(kl_prev_dict)

    combined_average = {}
    for k in scaled_disagreement:
        if k in scaled_kl:
            combined_average[k] = (scaled_disagreement[k] + scaled_kl[k]) / 2

    entropy_analyzer = EntropyByLayer(model_name, device=device)
    entropy_results_raw = entropy_analyzer.runStructuralAnalysis(prompt)

    entropy_dict = {f"layer_{res['layer']}": res['deltaH'] for res in entropy_results_raw}
    scaled_entropy = minMaxScale(entropy_dict)

    minmax_combined_all = {}
    for k in combined_average:
        if k in scaled_entropy:
            minmax_combined_all[k] = (combined_average[k] + scaled_entropy[k]) / 2

    sled_folder = f"results/{model_name}/{dataset}/sled_full"
    os.makedirs(sled_folder, exist_ok=True)
    sled_full_path = os.path.join(sled_folder, "sled_full_with_entropy.json")

    with open(sled_full_path, "w") as f:
        json.dump({
            "model": model_name,
            "dataset": dataset,
            "minmax_combined_all": minmax_combined_all
        }, f, indent=2)

    print(f"Results with SLED + entropy saved to {sled_full_path}")