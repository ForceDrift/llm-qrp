"""
AblationController — zeroes out individual transformer layers via forward hooks.

The previous implementation attempted to rebuild the model as an nn.Sequential,
which breaks transformer models because they have residual connections and
positional embeddings outside of the layer modules. Hook-based zeroing is
the correct approach.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


class AblationController:
    def __init__(self, modelId: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelId = modelId
        self.tokenizer = AutoTokenizer.from_pretrained(modelId)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelId, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.layerCount = len(self.model.model.layers)
        self._hooks: list = []

    # ------------------------------------------------------------------
    # Hook helpers
    # ------------------------------------------------------------------

    def _zero_hook(self, module, input, output):
        """Returns a zeroed tensor of the same shape as the layer output.

        Transformer layer outputs are typically a tuple where index 0 is the
        hidden state. We zero the hidden state and pass other elements through.
        """
        if isinstance(output, tuple):
            zeroed = torch.zeros_like(output[0])
            return (zeroed,) + output[1:]
        return torch.zeros_like(output)

    def ablate_layers(self, layer_indices: List[int]) -> None:
        """Register forward hooks that zero the output of the specified layers.

        Args:
            layer_indices: list of 0-indexed layer indices to ablate.
        """
        self.restore_layers()  # remove any existing hooks first
        for idx in layer_indices:
            if idx < 0 or idx >= self.layerCount:
                raise ValueError(
                    f"Layer index {idx} out of range [0, {self.layerCount - 1}]"
                )
            hook = self.model.model.layers[idx].register_forward_hook(self._zero_hook)
            self._hooks.append(hook)

    def restore_layers(self) -> None:
        """Remove all ablation hooks, restoring the model to its original state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Convenience: ablate by importance group
    # ------------------------------------------------------------------

    def ablate_top_fraction(self, sorted_layers: list, fraction: float = 0.2) -> List[int]:
        """Ablate the top `fraction` of layers by score (highest importance).

        Args:
            sorted_layers: list of (layer_idx, score) sorted ascending by score,
                           as produced by aggregate_scores.getSortedLayers().
            fraction: fraction of layers to ablate (default 0.2 = top 20%).

        Returns:
            List of ablated layer indices.
        """
        n = max(1, int(len(sorted_layers) * fraction))
        # sorted ascending → last n have the highest scores
        top_layers = [idx for idx, _ in sorted_layers[-n:]]
        self.ablate_layers(top_layers)
        return top_layers

    def ablate_bottom_fraction(self, sorted_layers: list, fraction: float = 0.2) -> List[int]:
        """Ablate the bottom `fraction` of layers by score (lowest importance).

        Args:
            sorted_layers: list of (layer_idx, score) sorted ascending by score.
            fraction: fraction of layers to ablate (default 0.2 = bottom 20%).

        Returns:
            List of ablated layer indices.
        """
        n = max(1, int(len(sorted_layers) * fraction))
        # sorted ascending → first n have the lowest scores
        bottom_layers = [idx for idx, _ in sorted_layers[:n]]
        self.ablate_layers(bottom_layers)
        return bottom_layers

    def __del__(self):
        self.restore_layers()
