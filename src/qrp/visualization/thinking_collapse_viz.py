import os
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from qrp.analysis.sled import SLED_Decoded
from qrp.analysis.ablation_controller import AblationController
from qrp.analysis.aggregate_scores import getSortedLayers

class ThinkingFlowVisualizer:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.ctrl = AblationController(model_name)
        self.ctrl.model.to(self.device)
        self.sled_model = SLED_Decoded(model_name, self.device)
        self.sled_model.model = self.ctrl.model  # Sync model
        self.num_layers = self.ctrl.layerCount

    def get_metrics_for_layers(self, prompt):
        """Returns SLED and KL metrics for every layer on a given prompt."""
        input_ids = self.ctrl.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        all_layers = list(range(self.num_layers))
        
        # 1. SLED Disagreement
        disagreement = self.sled_model.layer_disagreement(prompt, candidate_premature_layers=all_layers)
        sled_per_layer = np.zeros(self.num_layers)
        for d in disagreement:
            for l_idx, score in d.items():
                sled_per_layer[l_idx] += score.mean().item()
        sled_per_layer /= len(disagreement)
        
        # 2. Information Metrics
        with torch.no_grad():
            outputs = self.ctrl.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            final_logits = outputs.logits[:, -1, :].softmax(dim=-1)
            
            kl_seq = []
            kl_final = []
            entropies = []
            
            for i in range(1, len(hidden_states)):
                # Decoded logits for layer i
                curr_logits_raw = self.ctrl.model.lm_head(hidden_states[i][:, -1, :])
                curr_logits = curr_logits_raw.log_softmax(dim=-1)
                curr_probs = curr_logits_raw.softmax(dim=-1)
                
                # Entropy
                ent = -torch.sum(curr_probs * torch.log(curr_probs + 1e-9), dim=-1).item()
                entropies.append(ent)
                
                # KL seq
                prev_probs = self.ctrl.model.lm_head(hidden_states[i-1][:, -1, :]).softmax(dim=-1)
                kl_s = torch.nn.functional.kl_div(curr_logits, prev_probs, reduction="batchmean").item()
                kl_seq.append(kl_s)
                
                # KL final
                kl_f = torch.nn.functional.kl_div(curr_logits, final_logits, reduction="batchmean").item()
                kl_final.append(kl_f)
                
        return {
            "sled": sled_per_layer,
            "kl_seq": kl_seq,
            "kl_final": kl_final,
            "entropy": entropies
        }

    def run_comparison(self, prompt, target_thinking, target_redundant):
        print(f"--- Baseline ---")
        self.ctrl.restore_layers()
        baseline = self.get_metrics_for_layers(prompt)
        
        print(f"--- Ablating Thinking Layer {target_thinking} ---")
        self.ctrl.ablate_layers([target_thinking])
        thinking_ablated = self.get_metrics_for_layers(prompt)
        
        print(f"--- Ablating Redundant Layer {target_redundant} ---")
        self.ctrl.ablate_layers([target_redundant])
        redundant_ablated = self.get_metrics_for_layers(prompt)
        
        self.ctrl.restore_layers()
        return baseline, thinking_ablated, redundant_ablated

def plot_thinking_flow(baseline, thinking, redundant, t_idx, r_idx, out_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    layers = list(range(len(baseline["sled"])))
    
    # 1. Plot SLED Disagreement
    ax1.plot(layers, baseline["sled"], 'k-', label='Baseline (Intact)', linewidth=2)
    ax1.plot(layers, thinking["sled"], 'r--', label=f'Ablated Thinking (L{t_idx})', alpha=0.8)
    ax1.plot(layers, redundant["sled"], 'g:', label=f'Ablated Redundant (L{r_idx})', alpha=0.8)
    ax1.axvline(x=t_idx, color='r', linestyle=':', alpha=0.5, label='Ablation Point')
    ax1.set_title("Internal Disagreement Level (SLED Score)")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Disagreement Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot Entropy
    ax2.plot(layers, baseline["entropy"], 'k-', label='Baseline (Intact)', linewidth=2)
    ax2.plot(layers, thinking["entropy"], 'r--', label=f'Ablated Thinking (L{t_idx})', alpha=0.8)
    ax2.plot(layers, redundant["entropy"], 'g:', label=f'Ablated Redundant (L{r_idx})', alpha=0.8)
    ax2.axvline(x=t_idx, color='r', linestyle=':', alpha=0.5)
    ax2.set_title("Layer-wise Uncertainty (Information Entropy)")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Entropy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Plot KL Final
    ax3.plot(layers, baseline["kl_final"], 'k-', label='Baseline (Intact)', linewidth=2)
    ax3.plot(layers, thinking["kl_final"], 'r--', label=f'Ablated Thinking (L{t_idx})', alpha=0.8)
    ax3.plot(layers, redundant["kl_final"], 'g:', label=f'Ablated Redundant (L{r_idx})', alpha=0.8)
    ax3.axvline(x=t_idx, color='r', linestyle=':', alpha=0.5)
    ax3.set_title("Information Convergence Flow (KL-Div to Final Logits)")
    ax3.set_xlabel("Layer Index")
    ax3.set_ylabel("KL Divergence")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Graph saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--scores-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="If I have 3 apples and you take 1, how many do I have left?")
    parser.add_argument("--output-folder", type=str, default="results/reports")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Load scores to identify targets
    with open(args.scores_file, "r") as f:
        score_data = json.load(f)
    sorted_layers = getSortedLayers(score_data["layerAvgScores"])
    
    thinking_layer = sorted_layers[-1][0] # Highest
    redundant_layer = sorted_layers[0][0] # Lowest

    # Filter out Layer 31 for "thinking" because it's often trivial to ablate (final step)
    # Pick the next highest
    # thinking_layer = sorted_layers[-2][0]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    viz = ThinkingFlowVisualizer(args.model_name, device=device)
    b, t, r = viz.run_comparison(args.prompt, thinking_layer, redundant_layer)
    
    os.makedirs(args.output_folder, exist_ok=True)
    out_path = os.path.join(args.output_folder, "thinking_collapse.png")
    plot_thinking_flow(b, t, r, thinking_layer, redundant_layer, out_path)

if __name__ == "__main__":
    main()
