import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
class AblationController:
    def __init__(self, modelId):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(modelId)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelId, 
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.layerCount = len(self.model.model.layers)

    def ablateLayer(self, layer_indices):
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
            
        invalid_indices = [idx for idx in layer_indices if not (0 <= idx < self.layerCount)]
        if invalid_indices:
            raise ValueError(f"Invalid layer indices: {invalid_indices}")
            
        self._original_layers = list(self.model.model.layers)
        new_layers = nn.ModuleList([
            l for i, l in enumerate(self._original_layers) if i not in layer_indices
        ])
        self.model.model.layers = new_layers
        return self.model

    def restoreLayers(self):
        if hasattr(self, "_original_layers"):
            self.model.model.layers = self._original_layers

    

# to do: add benchmarking post-ablation for each layer pass, save to results
