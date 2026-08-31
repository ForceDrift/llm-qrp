import torch
from qrp.analysis.sled import SLED_Decoded

model = SLED_Decoded("Qwen/Qwen2.5-0.5B", "cuda")
prompt = "What is 2+2?"

logits_cpu, input_ids = model.compute_layer_logits_to_cpu(prompt)
print(f"Got {len(logits_cpu)} logits, shape: {logits_cpu[0].shape}")
print(f"Input IDs shape: {input_ids.shape}")

# Test KL
kl = model.kl_between_current_prev_from_cache(logits_cpu)
print(f"KL values count: {len(kl)}")

# Test disagreement
disagreement = model.layer_disagreement_from_cache(logits_cpu, input_ids, evolution_scale=5, candidate_premature_layers=None)
print(f"Disagreement entries: {len(disagreement)}")
if disagreement:
    print(f"First entry keys: {list(disagreement[0].keys())}")
else:
    print("ERROR: disagreement is empty!")

del model
torch.cuda.empty_cache()
