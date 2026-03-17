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

    def ablateLayer(self, layer_idx):
        if 0 <= layer_idx < self.layerCount:
            self._original_layers = self.model.model.layers
            new_layers = nn.ModuleList([l for i, l in enumerate(self._original_layers) if i != layer_idx])
            self.model.model.layers = new_layers
            return self.model
        else:
            raise ValueError(f"Invalid layer index: {layer_idx}")

    def restoreLayers(self):
        if hasattr(self, "_original_layers"):
            self.model.model.layers = self._original_layers

    

# to do: add benchmarking post-ablation for each layer pass, save to results
