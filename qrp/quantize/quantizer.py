import copy

import bitsandbytes as bnb
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from qrp.model_mapper import get_layer_structure, get_model_layers


class TargetedQuantizer:
    def __init__(self, model_id, device="auto"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.layers = get_model_layers(self.model)
        self.num_layers = len(self.layers)
        self.original_layers = {
            i: copy.deepcopy(layer) for i, layer in enumerate(self.layers)
        }

    def _quantize_module(self, parent, proj_names, quant_type="8bit"):
        if parent is None:
            return
        for name in proj_names:
            if not hasattr(parent, name):
                continue
            old_proj = getattr(parent, name)
            if quant_type == "8bit":
                new_proj = bnb.nn.Linear8bitLt(
                    old_proj.in_features, 
                    old_proj.out_features, 
                    bias=(old_proj.bias is not None),
                    has_fp16_weights=False
                )
            else:
                new_proj = bnb.nn.modules.Linear4bit(
                    old_proj.in_features, 
                    old_proj.out_features, 
                    bias=(old_proj.bias is not None), 
                    compute_dtype=torch.bfloat16,
                    quant_type="fp4"
                )
            if quant_type == "4bit":
                new_proj.weight = bnb.nn.modules.Params4bit(
                    old_proj.weight.data.clone(), 
                    requires_grad=False,
                    quant_type="fp4"
                )
            else:
                new_proj.weight.data.copy_(old_proj.weight.data)
            if old_proj.bias is not None:
                new_proj.bias = nn.Parameter(old_proj.bias.data.clone())
            new_proj.to(self.model.device)
            setattr(parent, name, new_proj)

    def quantize_layers(self, layer_configs):
        self.restore()
        for idx, quant_type in layer_configs.items():
            if 0 <= idx < self.num_layers:
                layer = self.layers[idx]
                (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layer)
                self._quantize_module(attn_parent, attn_projs, quant_type)
                self._quantize_module(mlp_parent, mlp_projs, quant_type)
        return self.model

    def restore(self):
        for i in range(self.num_layers):
            self.layers[i] = copy.deepcopy(self.original_layers[i])


