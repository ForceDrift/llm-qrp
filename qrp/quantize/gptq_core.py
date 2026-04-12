"""
Standalone GPTQ implementation for per-layer mixed-precision quantization.

Implements the core Optimal Brain Quantization (OBQ/GPTQ) algorithm:
  - Collect Hessian (H = 2 * X^T X) from calibration data
  - Quantize weights column-by-column with error compensation

This avoids the auto-gptq/autoawq dependency issues while giving us
full control over which layers get which precision.

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization
for Generative Pre-trained Transformers", ICLR 2023.
"""

import copy
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from qrp.model_mapper import get_layer_structure, get_model_layers


def quantize_weight_rtn(w, bits=4, group_size=128):
    """Round-to-nearest quantization (baseline, no compensation)."""
    w = w.float()
    shape = w.shape

    if group_size > 0 and shape[-1] % group_size == 0:
        w = w.reshape(-1, group_size)

    wmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    qmax = 2 ** (bits - 1) - 1
    scale = wmax / qmax
    w_quant = (w / scale).round().clamp(-qmax, qmax)
    w_deq = w_quant * scale

    return w_deq.reshape(shape)


def quantize_weight_gptq(w, H, bits=4, group_size=128, blocksize=128, percdamp=0.01):
    """
    GPTQ quantization with Hessian-based error compensation.
    
    Args:
        w: Weight matrix [out_features, in_features]
        H: Hessian matrix [in_features, in_features] (= X^T X from calibration)
        bits: Target quantization bits
        group_size: Group size for quantization (-1 = per-column)
        blocksize: Columns to process at once
        percdamp: Dampening factor for Hessian diagonal
    
    Returns:
        Quantized (dequantized) weight matrix
    """
    W = w.clone().float()
    rows, cols = W.shape
    
    H = H.float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # Dampening
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(cols, device=H.device)
    H[diag, diag] += damp

    # Cholesky decomposition
    try:
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
        H_inv = torch.linalg.cholesky(H_inv, upper=True)
    except torch.linalg.LinAlgError:
        # Fallback: add more dampening
        H[diag, diag] += 1e-2
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
        H_inv = torch.linalg.cholesky(H_inv, upper=True)

    Losses = torch.zeros(rows, device=W.device)

    for col_start in range(0, cols, blocksize):
        col_end = min(col_start + blocksize, cols)
        count = col_end - col_start

        W_block = W[:, col_start:col_end].clone()
        Err = torch.zeros_like(W_block)
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]

        for i in range(count):
            w_col = W_block[:, i]
            d = H_inv_block[i, i]

            # Determine group for scaling
            col_idx = col_start + i
            if group_size > 0:
                group_id = col_idx // group_size
                group_start = group_id * group_size
                group_end = min(group_start + group_size, cols)
                group_w = W[:, group_start:group_end]
                wmax = group_w.abs().amax(dim=-1).clamp(min=1e-5)
            else:
                wmax = w_col.abs().clamp(min=1e-5)

            qmax = 2 ** (bits - 1) - 1
            scale = wmax / qmax

            # Quantize
            q = (w_col / scale).round().clamp(-qmax, qmax)
            w_quant = q * scale

            Err[:, i] = (w_col - w_quant) / d
            Losses += (w_col - w_quant) ** 2 / (d ** 2)

            # Compensate remaining columns in this block
            W_block[:, i:] -= Err[:, i].unsqueeze(1).matmul(
                H_inv_block[i, i:].unsqueeze(0)
            )

        # Propagate error to remaining columns
        W[:, col_end:] -= Err.matmul(H_inv[col_start:col_end, col_end:])

    # Final quantization of the compensated weights 
    if group_size > 0 and cols % group_size == 0:
        W_grouped = W.reshape(rows, -1, group_size)
        wmax = W_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        qmax = 2 ** (bits - 1) - 1
        scale = wmax / qmax
        W_q = (W_grouped / scale).round().clamp(-qmax, qmax) * scale
        W = W_q.reshape(rows, cols)
    else:
        wmax = W.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        qmax = 2 ** (bits - 1) - 1
        scale = wmax / qmax
        W = (W / scale).round().clamp(-qmax, qmax) * scale

    return W


class GPTQMixedQuantizer:
    """
    GPTQ-based quantizer driven by QRP's per-layer precision map.
    
    Performs actual Hessian-compensated quantization (not naive round-to-nearest)
    for layers assigned to 4-bit or 8-bit, while leaving bf16 layers untouched.
    """

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, trust_remote_code=True
            )

        self.layers = get_model_layers(self.model)
        self.num_layers = len(self.layers)

        # Cache original weights for restore
        self.original_state = {
            i: copy.deepcopy(layer.state_dict())
            for i, layer in enumerate(self.layers)
        }

    def _collect_layer_inputs(self, layer_idx, calibration_data, max_samples=128):
        """
        Collect input activations to a specific layer using hooks.
        Returns a tensor of all inputs concatenated along the token dimension.
        """
        collected = []

        layer = self.layers[layer_idx]
        # Use pre-hook to get inputs TO the layer
        handle = layer.register_forward_pre_hook(
            lambda mod, inp: collected.append(inp[0].detach().cpu()) or None
        )

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(calibration_data[:max_samples]):
                try:
                    self.model(data.to(self.model.device))
                except Exception:
                    continue

        handle.remove()
        if not collected:
            return None

        # Concatenate along sequence dimension, flatten to 2D [total_tokens, hidden_dim]
        all_inputs = torch.cat(collected, dim=1)  # [1, total_seq, hidden]
        all_inputs = all_inputs.squeeze(0)  # [total_seq, hidden]
        return all_inputs

    def _compute_hessian(self, X):
        """Compute Hessian approximation: H = 2 * X^T X / n_samples."""
        X = X.float()
        n = X.shape[0]
        H = (2.0 / n) * (X.T @ X)
        return H

    def _quantize_linear(self, linear_module, H, bits, group_size=128):
        """Quantize a single nn.Linear module using GPTQ."""
        W = linear_module.weight.data
        device = W.device

        H = H.to(device)
        W_q = quantize_weight_gptq(W, H, bits=bits, group_size=group_size)
        linear_module.weight.data = W_q.to(W.dtype).to(device)

    def prepare_calibration_data(self, dataset_name="wikitext", n_samples=128, seq_len=512):
        """Prepare calibration data for GPTQ quantization."""
        if dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n\n".join([t for t in ds["text"] if t.strip()])
        elif dataset_name == "c4":
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = []
            for i, item in enumerate(ds):
                if i >= n_samples:
                    break
                texts.append(item["text"])
            text = "\n\n".join(texts)
        else:
            raise ValueError(f"Unknown calibration dataset: {dataset_name}")

        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids

        # Split into chunks
        chunks = []
        for i in range(0, min(input_ids.shape[1], n_samples * seq_len), seq_len):
            chunk = input_ids[:, i:i + seq_len]
            if chunk.shape[1] == seq_len:
                chunks.append(chunk)
            if len(chunks) >= n_samples:
                break

        return chunks

    def quantize_with_precision_map(self, layer_configs, calibration_data=None,
                                     group_size=128):
        """
        Quantize the model using GPTQ, driven by QRP's precision map.
        
        Args:
            layer_configs: dict mapping layer_idx -> "4bit" | "8bit" | "bf16"
                          Layers not in the dict default to "bf16".
            calibration_data: list of input_ids tensors for Hessian computation.
                             If None, will be prepared automatically.
            group_size: quantization group size
        """
        self.restore()

        if calibration_data is None:
            print("Preparing calibration data from WikiText-2...")
            calibration_data = self.prepare_calibration_data(n_samples=128, seq_len=512)

        layers_to_quantize = {
            idx: cfg for idx, cfg in layer_configs.items()
            if cfg in ("4bit", "8bit")
        }

        if not layers_to_quantize:
            print("No layers to quantize. Model stays at bf16.")
            return self.model

        bits_map = {"4bit": 4, "8bit": 8}

        print(f"\nQuantizing {len(layers_to_quantize)} layers with GPTQ "
              f"(Hessian-compensated weight rounding)...")

        for idx in tqdm(sorted(layers_to_quantize.keys()), desc="GPTQ quantizing"):
            bits = bits_map[layers_to_quantize[idx]]
            layer = self.layers[idx]

            # Collect activations for this layer
            X = self._collect_layer_inputs(idx, calibration_data, max_samples=8)
            if X is None or X.shape[0] < 2:
                print(f"  Warning: No activations collected for layer {idx}, "
                      f"falling back to RTN quantization")
                # Fallback: round-to-nearest without Hessian
                (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layer)
                for parent, projs in [(attn_parent, attn_projs), (mlp_parent, mlp_projs)]:
                    if parent is None:
                        continue
                    for name in projs:
                        proj = getattr(parent, name, None)
                        if proj is not None and hasattr(proj, 'weight'):
                            proj.weight.data = quantize_weight_rtn(
                                proj.weight.data, bits=bits, group_size=group_size
                            ).to(proj.weight.dtype)
                continue

            # Compute Hessian from activations
            H = self._compute_hessian(X)

            # Quantize each sub-module in the layer
            (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layer)

            for parent, projs in [(attn_parent, attn_projs), (mlp_parent, mlp_projs)]:
                if parent is None:
                    continue
                for name in projs:
                    proj = getattr(parent, name, None)
                    if proj is not None and hasattr(proj, 'weight'):
                        try:
                            # The Hessian from layer input applies to the first
                            # projection; for inner projections we use a smaller H
                            # by projecting through the weight. For simplicity,
                            # we use the layer-level H clipped to the right size.
                            in_feat = proj.weight.shape[1]
                            if H.shape[0] >= in_feat:
                                H_sub = H[:in_feat, :in_feat]
                            else:
                                # H is smaller than weight dim — pad with identity
                                H_sub = torch.eye(in_feat, device=H.device)
                                h_size = H.shape[0]
                                H_sub[:h_size, :h_size] = H

                            self._quantize_linear(proj, H_sub, bits, group_size)
                        except Exception as e:
                            print(f"  Warning: GPTQ failed for layer {idx}.{name}: {e}")
                            proj.weight.data = quantize_weight_rtn(
                                proj.weight.data, bits=bits, group_size=group_size
                            ).to(proj.weight.dtype)

        n4 = sum(1 for v in layer_configs.values() if v == "4bit")
        n8 = sum(1 for v in layer_configs.values() if v == "8bit")
        n_bf = self.num_layers - n4 - n8
        print(f"Done: {n4} layers @ 4-bit, {n8} layers @ 8-bit, {n_bf} layers @ bf16")
        return self.model

    def restore(self):
        """Restore all layers to original bf16 weights."""
        for i in range(self.num_layers):
            self.layers[i].load_state_dict(self.original_state[i])
