import copy

import bitsandbytes as bnb
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from qrp.model_mapper import get_layer_structure, get_model_layers
from qrp.quantize.integer_quant import OutlierProtectedLinear, UniformIntLinear


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

    def quantize_components(self, component_configs, outlier_channels=None):
        """Quantize at sub-component granularity.

        ``component_configs`` maps ``"<layer_idx>.<component>"`` (e.g.
        ``"3.attn"``) to a precision in
        ``{"2bit", "3bit", "4bit", "8bit", "16bit", "bf16"}``.  Attention and
        MLP of the same layer can receive *different* precisions.

        ``outlier_channels`` optionally maps the same component ids to the
        top-0.1% salient channel indices captured during profiling; those
        channels stay unquantized (BF16 ``W_fp16``) while the remaining 99.9%
        are quantized to the requested low-bit precision.
        """
        self.restore()
        for key, quant_type in component_configs.items():
            layer_str, c = key.rsplit(".", 1)
            layer_idx = int(layer_str)
            if not (0 <= layer_idx < self.num_layers) or c not in ("attn", "mlp"):
                continue
            normalized = {"2bit": "2bit", "3bit": "3bit", "4bit": "4bit",
                          "6bit": "6bit", "8bit": "8bit",
                          "16bit": "skip", "bf16": "skip"}.get(quant_type)
            if normalized == "skip":
                continue
            layer = self.layers[layer_idx]
            (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layer)
            parent = attn_parent if c == "attn" else mlp_parent
            projs = attn_projs if c == "attn" else mlp_projs
            protected = (outlier_channels or {}).get(key, [])
            self._quantize_projection(parent, projs, normalized, protected)
        return self.model

    def _quantize_projection(self, parent, proj_names, quant_type, outlier_cols):
        """Quantize one sub-component's projections to ``quant_type``.

        Outlier-protected / ultra-low-bit (2/3) projections use the in-repo
        integer quantizer; 4/8-bit use ``bitsandbytes``.
        """
        if parent is None:
            return
        for name in proj_names:
            if not hasattr(parent, name):
                continue
            old_proj = getattr(parent, name)
            if outlier_cols:
                new_proj = OutlierProtectedLinear(old_proj, outlier_cols, int(quant_type[0]))
            elif quant_type in ("2bit", "3bit", "6bit"):
                new_proj = UniformIntLinear(old_proj, int(quant_type[0]))
            elif quant_type == "4bit":
                new_proj = bnb.nn.modules.Linear4bit(
                    old_proj.in_features, old_proj.out_features,
                    bias=(old_proj.bias is not None),
                    compute_dtype=torch.bfloat16, quant_type="fp4",
                )
                new_proj.weight = bnb.nn.modules.Params4bit(
                    old_proj.weight.data.clone(), requires_grad=False, quant_type="fp4"
                )
                if old_proj.bias is not None:
                    new_proj.bias = nn.Parameter(old_proj.bias.data.clone())
                new_proj.to(self.model.device)
                setattr(parent, name, new_proj)
                continue
            elif quant_type == "8bit":
                new_proj = bnb.nn.Linear8bitLt(
                    old_proj.in_features, old_proj.out_features,
                    bias=(old_proj.bias is not None), has_fp16_weights=False,
                )
                new_proj.weight.data.copy_(old_proj.weight.data)
                if old_proj.bias is not None:
                    new_proj.bias = nn.Parameter(old_proj.bias.data.clone())
                new_proj.to(self.model.device)
                setattr(parent, name, new_proj)
                continue
            else:
                continue
            if old_proj.bias is not None:
                new_proj.bias = nn.Parameter(old_proj.bias.data.clone())
            new_proj.to(self.model.device)
            setattr(parent, name, new_proj)

    def restore(self):
        for i in range(self.num_layers):
            self.layers[i] = copy.deepcopy(self.original_layers[i])


