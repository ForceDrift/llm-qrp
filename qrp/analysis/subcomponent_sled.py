"""CoT-masked, sub-component Self-Logits Evolution (SLED) profiling.

Implements the SLED formulation at the *sub-component* granularity.  Each
decoder layer ``l`` is decomposed into two components ``c``:

  * ``attn`` -- the self-attention sub-block residual output
    ``h_{l,t}^{attn} = h_{l-1,t} + Attn(h_{l-1,t})``
  * ``mlp``  -- the MLP sub-block residual output
    ``h_{l,t}^{mlp}  = h_{l,t}^{attn} + MLP(Norm(h_{l,t}^{attn}))``

Candidate logits are obtained with a logit-lens projection of each
sub-component residual onto the vocabulary space:

    l_{l,t}^{attn} = W_lm_head . Norm(h_{l,t}^{attn})
    l_{l,t}^{mlp}  = W_lm_head . Norm(h_{l,t}^{mlp})

Scoring is restricted to *intermediate Chain-of-Thought (CoT) reasoning token*
positions ``t in T_CoT``; prompt and template tokens are ignored.  Two signals
are produced per (layer, component):

  1. The sub-component SLED score -- average cosine similarity between the
     evolution gradient ``g_{l,t}^{(c)} = softmax(l_{l,t}^{(c)}) - 1_{argmax l_{M,t}}``
     and the logit divergence
     ``delta_{l,t}^{(c)} = logsoftmax(l_{l,t}^{(c)}) - logsoftmax(l_{M,t})``:

        S_SLED(l, c) = 1/|T_CoT| * sum_{t in T_CoT} cos(g_{l,t}^{(c)}, delta_{l,t}^{(c)})

  2. The Information-Bottleneck Convergence Velocity -- how aggressively the
     sub-block reduces prediction uncertainty over CoT tokens.  Let
     ``P_{l,t}^{(c)} = softmax(l_{l,t}^{(c)})`` be the projected vocabulary
     distribution at sub-component input (``P_in``) and output (``P_out``)
     residuals, with Shannon entropy
     ``H(P) = -sum_v P(v) log P(v)``.  The entropy velocity is the average
     absolute entropy transition across the sub-block:

        DeltaH(l, c) = 1/|T_CoT| * sum_{t in T_CoT} |H(P_{l,t}^{in}) - H(P_{l,t}^{out})|

     This is purely a forward-pass measurement -- no synthetic 4-bit injection
     noise is involved.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from qrp.model_mapper import get_layer_structure, get_model_layers

if TYPE_CHECKING:
    from torch import Tensor

COMPONENTS = ("attn", "mlp")


class SubComponentSLED:
    """CoT-masked sub-component SLED scorer.

    A single forward pass on ``input_ids`` returns per-(layer, component)
    SLED scores ``S_SLED(l, c)`` averaged over the CoT reasoning positions.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(device)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.layers = get_model_layers(self.model)
        self.num_layers = len(self.layers)
        self.lm_head = self.model.get_output_embeddings()
        # Final (post-layer) normalisation applied before the LM head.  Applying
        # the same Norm inside the logit lens keeps the projection on the same
        # scale as the real output logits.
        self.final_norm = getattr(self.model, "norm", None)

    # ------------------------------------------------------------------ #
    # Logit-lens projection
    # ------------------------------------------------------------------ #
    def logit_lens(self, hidden: Tensor) -> Tensor:
        h = hidden
        if self.final_norm is not None:
            h = self.final_norm(h)
        # Cast to the LM-head weight dtype so float32 residuals work with a
        # bf16/fp16 model (avoids "mat1 and mat2 must have the same dtype").
        if h.dtype != self.lm_head.weight.dtype:
            h = h.to(self.lm_head.weight.dtype)
        return self.lm_head(h)

    # ------------------------------------------------------------------ #
    # Sub-component residual capture
    # ------------------------------------------------------------------ #
    def _capture_subcomponent_residuals(self, input_ids: Tensor):
        """Return per-layer sub-block input/output residuals.

        Returns ``(attn_out, mlp_out, layer_in, mature_logits, num_tokens)``:

          * ``layer_in[l]`` -- residual before the attention sub-block (the
            sub-component *input* for ``attn``);
          * ``attn_out[l]`` -- residual after the attention add
            ``h_in + Attn(Norm(h_in))`` (``attn`` output, ``mlp`` input);
          * ``mlp_out[l]``  -- layer output residual ``hidden_states[l + 1]``.

        Residuals follow the Llama-style post-norm block layout.
        """
        layer_inputs: dict[int, Tensor] = {}
        attn_outputs: dict[int, Tensor] = {}

        def _layer_in_hook(layer_idx: int):
            def hook(_module, args, _output):
                layer_inputs[layer_idx] = args[0].detach()

            return hook

        def _attn_out_hook(layer_idx: int, module):
            def hook(_module, _args, output):
                out = output
                if isinstance(out, (tuple, list)):
                    out = out[0]
                attn_outputs[layer_idx] = out.detach()

            return hook

        hooks = []
        for l in range(self.num_layers):
            layer = self.layers[l]
            hooks.append(
                layer.register_forward_hook(_layer_in_hook(l))
            )
            (attn_parent, _attn_projs), (_mlp_parent, _mlp_projs) = get_layer_structure(layer)
            if attn_parent is not None:
                hooks.append(attn_parent.register_forward_hook(_attn_out_hook(l, attn_parent)))

        try:
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
        finally:
            for hook in hooks:
                hook.remove()

        hidden_states = outputs.hidden_states  # [0 .. num_layers] -> h_mlp[l - 1]
        mature_logits = outputs.logits
        num_tokens = input_ids.shape[-1]

        attn_residuals: dict[int, Tensor] = {}
        mlp_residuals: dict[int, Tensor] = {}
        for l in range(self.num_layers):
            h_in = layer_inputs.get(l)
            attn_out = attn_outputs.get(l)
            if h_in is None or attn_out is None:
                raise RuntimeError(
                    f"Failed to capture sub-component residuals for layer {l} "
                    f"(layer_input={h_in is not None}, attn_out={attn_out is not None})."
                )
            attn_residuals[l] = (h_in + attn_out).float()
            mlp_residuals[l] = hidden_states[l + 1].float()

        return attn_residuals, mlp_residuals, layer_inputs, mature_logits, num_tokens

    # ------------------------------------------------------------------ #
    # Vocabulary masking
    # ------------------------------------------------------------------ #
    @staticmethod
    def relative_top_mask(mature_logits: Tensor, relative_top: float = 0.1) -> Tensor:
        """Mask low-relative-log-probability vocabulary noise for one position.

        Returns a boolean mask, True where a token is rejected (kept out of the
        cosine-similarity computation), mirroring the SLED reference filter.
        """
        scores_prob = mature_logits.log_softmax(dim=-1)
        top_logit = scores_prob.max(dim=-1).values
        thresh = torch.min(top_logit, top_logit + math.log(relative_top))
        return scores_prob < thresh.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    # Score computation
    # ------------------------------------------------------------------ #
    def _component_scores(self, candidate: Tensor, mature: Tensor, mask: Tensor) -> Tensor:
        """Cosine similarity between masked evolution gradient and divergence.

        g = softmax(l_cand) - 1_{argmax l_M}          (evolution gradient)
        delta = logsoftmax(l_cand) - logsoftmax(l_M)  (mature logit divergence)
        """
        eff = ~mask  # vocabulary entries kept after relative-top filtering
        g = candidate.softmax(dim=-1)
        y_hat = mature.argmax(dim=-1)
        g = g - F.one_hot(y_hat, num_classes=g.shape[-1]).to(g.dtype)
        delta = F.log_softmax(candidate.float(), dim=-1) - F.log_softmax(mature.float(), dim=-1)
        g = (g * eff).float()
        delta = (delta * eff).float()
        # [B, V] -> cosine similarity over the vocabulary dimension
        cos = F.cosine_similarity(g, delta, dim=-1)
        return torch.nan_to_num(cos, nan=0.0)

    def score(self, input_ids: Tensor, cot_start: Optional[int] = None) -> dict[str, float]:
        """Compute S_SLED(l, c) for every (layer, component).

        ``cot_start`` is the token index where the CoT reasoning region begins;
        only positions ``t in [cot_start, seq_len)`` contribute.  When
        ``cot_start`` is None the *entire* sequence contributes (no masking).
        """
        res = self._capture_subcomponent_residuals(input_ids)
        return self._compute_sled(res, cot_start)

    def _compute_sled(self, res, cot_start: Optional[int]) -> dict[str, float]:
        attn_residuals, mlp_residuals, _layer_in, mature_logits, num_tokens = res
        start = cot_start if cot_start is not None else 0
        num_positions = max(1, num_tokens - start)
        positions = range(start, num_tokens)

        scores: dict[str, list[float]] = {}
        for l in range(self.num_layers):
            for c in COMPONENTS:
                scores[f"{l}.{c}"] = []

        mature = mature_logits.float()
        for t in positions:
            mask = self.relative_top_mask(mature[:, t, :])
            mature_t = mature[:, t, :]
            for l in range(self.num_layers):
                for c in COMPONENTS:
                    res = attn_residuals[l] if c == "attn" else mlp_residuals[l]
                    cand = self.logit_lens(res[:, t, :].unsqueeze(1)).squeeze(1).float()
                    per_token = self._component_scores(cand, mature_t, mask)  # [B]
                    scores[f"{l}.{c}"].append(float(per_token.mean()))

        return {key: sum(v) / num_positions for key, v in scores.items()}

    @staticmethod
    def _shannon_entropy(logits: Tensor) -> Tensor:
        """Per-position Shannon entropy ``H(P)`` over the vocabulary, [B, S]."""
        probs = logits.softmax(dim=-1)
        logp = torch.log(probs.clamp_min(1e-9))
        return -(probs * logp).sum(dim=-1)

    def entropy_velocity(self, input_ids: Tensor, cot_start: Optional[int] = None) -> dict[str, float]:
        """Information-Bottleneck Convergence Velocity per (layer, component).

        ``DeltaH(l, c)`` is the average absolute entropy transition of the
        projected vocabulary distribution across the sub-block, restricted to
        CoT reasoning positions:

            DeltaH(l, c) = 1/|T_CoT| * sum_{t} |H(P_{l,t}^{in}) - H(P_{l,t}^{out})|

        High velocity means the sub-component aggressively collapses prediction
        uncertainty, i.e. it acts as an information bottleneck that must be
        preserved at high precision.  A single forward pass; no quantization
        injection.
        """
        attn_out, mlp_out, layer_in, _mature, num_tokens = self._capture_subcomponent_residuals(input_ids)
        return self._compute_velocity((attn_out, mlp_out, layer_in, _mature, num_tokens), cot_start)

    def _compute_velocity(self, res, cot_start: Optional[int]) -> dict[str, float]:
        attn_out, mlp_out, layer_in, _mature, num_tokens = res
        start = cot_start if cot_start is not None else 0
        sl = slice(start, num_tokens)

        results: dict[str, list[float]] = {f"{l}.{c}": [] for l in range(self.num_layers) for c in COMPONENTS}
        for l in range(self.num_layers):
            h_in = self.logit_lens(layer_in[l]).float()
            h_attn = self.logit_lens(attn_out[l]).float()
            attn_in_h = self._shannon_entropy(h_in)[:, sl]
            attn_out_h = self._shannon_entropy(h_attn)[:, sl]
            results[f"{l}.attn"] = (attn_in_h - attn_out_h).abs().mean(dim=-1).item()

        for l in range(self.num_layers):
            h_attn = self.logit_lens(attn_out[l]).float()
            h_mlp = self.logit_lens(mlp_out[l]).float()
            mlp_in_h = self._shannon_entropy(h_attn)[:, sl]
            mlp_out_h = self._shannon_entropy(h_mlp)[:, sl]
            results[f"{l}.mlp"] = (mlp_in_h - mlp_out_h).abs().mean(dim=-1).item()

        return {f"{l}.{c}": float(results[f"{l}.{c}"]) for l in range(self.num_layers) for c in COMPONENTS}

    # ------------------------------------------------------------------ #
    # Salient outlier channel protection (0.1% per sub-matrix)
    # ------------------------------------------------------------------ #
    OUTLIER_SHARE = 0.001

    @staticmethod
    def _channel_salience(activations: Tensor) -> Tensor:
        """Mean activation magnitude per channel over batch, CoT, and tokens."""
        return activations.float().abs().mean(dim=(0, 1))

    @staticmethod
    def _top_channels(salience: Tensor, k: int) -> list[int]:
        k = max(0, min(k, salience.numel()))
        if k == 0:
            return []
        return torch.sort(salience, descending=True).indices[:k].cpu().tolist()

    def _compute_salient(self, res, cot_start: Optional[int], k: int) -> dict[str, list[int]]:
        attn_out, mlp_out, layer_in, _mature, num_tokens = res
        start = cot_start if cot_start is not None else 0
        sl = slice(start, num_tokens)
        channels: dict[str, list[int]] = {}
        for l in range(self.num_layers):
            attn_act = self._channel_salience(layer_in[l][:, sl, :])
            channels[f"{l}.attn"] = self._top_channels(attn_act, k)
            mlp_act = self._channel_salience(attn_out[l][:, sl, :])
            channels[f"{l}.mlp"] = self._top_channels(mlp_act, k)
        return channels

    def salient_channels(self, input_ids: Tensor, cot_start: Optional[int] = None,
                         k: Optional[int] = None) -> dict[str, list[int]]:
        """Top-``k`` highest-activation channels per (layer, component).

        ``k`` defaults to the top 0.1% of the hidden dimension (at least 1).
        These channels are kept unquantized (BF16 sparse ``W_fp16``) during
        quantization while the remaining 99.9% are quantized to low bits.
        """
        res = self._capture_subcomponent_residuals(input_ids)
        hidden_dim = res[2][0].shape[-1]
        if k is None:
            k = max(1, math.ceil(self.OUTLIER_SHARE * hidden_dim))
        return self._compute_salient(res, cot_start, k)

    def profile(self, input_ids: Tensor, cot_start: Optional[int] = None,
                outlier_share: Optional[float] = None,
                channels_per_component: Optional[int] = None) -> dict:
        """All sub-component signals from a *single* forward pass.

        Returns ``{"sled", "entropy", "outlier_channels"}`` for CoT-masked
        positions, plus the number of protected channels per component.
        """
        res = self._capture_subcomponent_residuals(input_ids)
        hidden_dim = res[2][0].shape[-1]
        if channels_per_component is None:
            share = self.OUTLIER_SHARE if outlier_share is None else float(outlier_share)
            channels_per_component = max(1, math.ceil(share * hidden_dim))
        return {
            "sled": self._compute_sled(res, cot_start),
            "entropy": self._compute_velocity(res, cot_start),
            "outlier_channels": self._compute_salient(res, cot_start, channels_per_component),
            "channels_per_component": channels_per_component,
        }