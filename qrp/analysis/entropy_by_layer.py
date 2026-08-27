"""Synthetic-injection entropy sensitivity (4-bit perturbation on last logits).

DEPRECATED as the framework's entropy signal.  SECTION 3.2 of the evolved
methodology redefines entropy as the *Information-Bottleneck Convergence
Velocity* -- a forward-only measurement of the average absolute vocabulary
entropy transition across each sub-block over CoT tokens (implemented in
``qrp.analysis.subcomponent_sled.SubComponentSLED.entropy_velocity``).  The
injection-based ``DeltaH`` below is retained only for the whole-layer legacy
pipeline (``runStructuralAnalysis``) and for ablation comparisons of the two
entropy formulations.
"""

import copy

import bitsandbytes
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from qrp.model_mapper import get_layer_structure, get_model_layers

# Sub-component keys, matching ``subcomponent_sled.COMPONENTS``.
COMPONENTS = ("attn", "mlp")


class EntropyByLayer:
    def __init__(self, modelId, device="auto"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(modelId)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelId,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if device != "auto":
            self.model.to(device)
        self.layers = get_model_layers(self.model)
        self.numLayers = len(self.layers)

    def _calculateMetrics(self, currentLogits, baselineProbs):
        logits32 = currentLogits.detach().to(torch.float32)
        probs32 = baselineProbs.detach().to(torch.float32)
        currentLogProbs = F.log_softmax(logits32, dim=-1)
        klDiv = F.kl_div(currentLogProbs, probs32, reduction="batchmean").item()
        softmaxProbs = F.softmax(logits32, dim=-1)
        entropy = -torch.sum(softmaxProbs * torch.log(softmaxProbs + 1e-9), dim=-1).item()
        return klDiv, entropy

    def _swap_to_4bit(self, parent, projNames):
        if parent is None:
            return
        for x in projNames:
            if not hasattr(parent, x):
                continue
            oldProj = getattr(parent, x)
            newProj = bitsandbytes.nn.modules.Linear4bit(
                oldProj.in_features,
                oldProj.out_features,
                bias=(oldProj.bias is not None),
                compute_dtype=torch.float16,
            )
            newProj.weight = bitsandbytes.nn.modules.Params4bit(
                oldProj.weight.data.clone(),
                requires_grad=False,
            )
            if oldProj.bias is not None:
                newProj.bias = torch.nn.Parameter(oldProj.bias.data.clone())
            newProj.to(self.model.device)
            setattr(parent, x, newProj)

    def _run_q4_entropy(self, inputs, baselineProbs, hBase, layerIdx, moduleType, parent, projNames):
        layerBackup = copy.deepcopy(self.layers[layerIdx])
        self._swap_to_4bit(parent, projNames)
        klVal, deltaH = float("nan"), float("nan")
        try:
            with torch.no_grad():
                currentOutput = self.model(**inputs)
                currentLogits = currentOutput.logits[:, -1, :]
                klVal, hVal = self._calculateMetrics(currentLogits, baselineProbs)
            deltaH = abs(hVal - hBase)
        finally:
            self.layers[layerIdx] = layerBackup
        return {"layer": layerIdx, "component": moduleType, "kl": klVal, "deltaH": deltaH}

    def runStructuralAnalysis(self, prompt):
        """[legacy] Whole-layer structural entropy sensitivity (attn + MLP together).

        Superseded by ``SubComponentSLED.entropy_velocity`` in the sub-component
        pipeline; kept only for the whole-layer path and ablations.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            baselineOutput = self.model(**inputs)
            baselineLogits = baselineOutput.logits[:, -1, :]
            baselineProbs = F.softmax(baselineLogits.float(), dim=-1)
            hBase = -torch.sum(baselineProbs * torch.log(baselineProbs + 1e-9), dim=-1).item()

        levResults = []
        for i in tqdm(range(self.numLayers)):
            targetLayer = self.layers[i]
            layerBackup = copy.deepcopy(targetLayer)
            (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(targetLayer)
            targetModules = {
                "attn": (attn_parent, attn_projs),
                "mlp": (mlp_parent, mlp_projs),
            }

            for moduleType, (parent, projNames) in targetModules.items():
                if parent is None:
                    continue
                self._swap_to_4bit(parent, projNames)

            with torch.no_grad():
                currentOutput = self.model(**inputs)
                currentLogits = currentOutput.logits[:, -1, :]
                klVal, hVal = self._calculateMetrics(currentLogits, baselineProbs)

            levResults.append({
                "layer": i,
                "kl": klVal,
                "deltaH": abs(hVal - hBase),
            })
            self.layers[i] = layerBackup

        return levResults

    def runSubcomponentAnalysis(self, prompt):
        """[legacy] Synthetic-injection sub-component entropy, kept for ablation.

        DEPRECATED: the Section 3.2 entropy signal is now the forward-only
        Information-Bottleneck Convergence Velocity
        (``SubComponentSLED.entropy_velocity``).  This injection variant is
        retained solely for ablation work and is no longer invoked by the
        profiling pipeline.

        Attention/MLP projections are quantized to 4 bits *independently*; the
        Shannon-entropy perturbation ``DeltaH = |H(P_base) - H(P_q)|`` is
        reported per sub-component.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            baselineOutput = self.model(**inputs)
            baselineLogits = baselineOutput.logits[:, -1, :]
            baselineProbs = F.softmax(baselineLogits.float(), dim=-1)
            hBase = -torch.sum(baselineProbs * torch.log(baselineProbs + 1e-9), dim=-1).item()

        subResults = []
        for i in tqdm(range(self.numLayers)):
            targetLayer = self.layers[i]
            (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(targetLayer)
            for moduleType, parent, projNames in (("attn", attn_parent, attn_projs), ("mlp", mlp_parent, mlp_projs)):
                if parent is None:
                    continue
                subResults.append(
                    self._run_q4_entropy(inputs, baselineProbs, hBase, i, moduleType, parent, projNames)
                )
        return subResults