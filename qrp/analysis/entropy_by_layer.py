# import copy
# import torch
# import torch.nn.functional as F
# import bitsandbytes
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

# class EntropyByLayer:
#     def __init__(self, modelId, device="auto"):
#         self.device = device
#         self.tokenizer = AutoTokenizer.from_pretrained(modelId)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             modelId, 
#             torch_dtype=torch.bfloat16, 
#             device_map=device
#         )
#         self.numLayers = len(self.model.model.layers)

#     def _calculateMetrics(self, currentLogits, baselineProbs):
#         logits32 = currentLogits.detach().to(torch.float32)
#         probs32 = baselineProbs.detach().to(torch.float32)

#         currentLogProbs = F.log_softmax(logits32, dim=-1)
#         klDiv = F.kl_div(currentLogProbs, probs32, reduction='batchmean').item()
        
#         softmaxProbs = F.softmax(logits32, dim=-1)
#         entropy = -torch.sum(softmaxProbs * torch.log(softmaxProbs + 1e-9), dim=-1).item()
        
#         return klDiv, entropy

#     def runStructuralAnalysis(self, prompt):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
#         with torch.no_grad():
#             baselineOutput = self.model(**inputs)
#             baselineLogits = baselineOutput.logits[:, -1, :]
#             baselineProbs = F.softmax(baselineLogits.float(), dim=-1)
#             hBase = -torch.sum(baselineProbs * torch.log(baselineProbs + 1e-9), dim=-1).item()

#         levResults = []
        
#         for i in tqdm(range(self.numLayers)):
#             targetLayer = self.model.model.layers[i]
#             originalState = copy.deepcopy(targetLayer.state_dict())
            
#             targetModules = {
#                 "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], 
#                 "mlp": ["gate_proj", "up_proj", "down_proj"]
#             }

#             for moduleType, projNames in targetModules.items():
#                 parent = targetLayer.self_attn if moduleType == "attn" else targetLayer.mlp
#                 for x in projNames:
#                     oldProj = getattr(parent, x)
#                     newProj = bitsandbytes.nn.modules.Linear4bit(
#                         oldProj.in_features, 
#                         oldProj.out_features, 
#                         bias=(oldProj.bias is not None), 
#                         compute_dtype=torch.float16
#                     )
#                     newWeight = bitsandbytes.nn.modules.Params4bit(
#                         oldProj.weight.data.clone(), 
#                         requires_grad=False
#                     )
#                     newProj.weight = newWeight
#                     if oldProj.bias is not None:
#                         newProj.bias = torch.nn.Parameter(oldProj.bias.data.clone())
                    
#                     newProj.to(self.model.device)
#                     setattr(parent, x, newProj)

#             with torch.no_grad():
#                 currentOutput = self.model(**inputs)
#                 currentLogits = currentOutput.logits[:, -1, :]
#                 klVal, hVal = self._calculateMetrics(currentLogits, baselineProbs)
            
#             levResults.append({
#                 "layer": i, 
#                 "kl": klVal, 
#                 "deltaH": abs(hVal - hBase)
#             })

#             targetLayer.load_state_dict(originalState)

#         return levResults

import copy
import torch
import torch.nn.functional as F
import bitsandbytes
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class EntropyByLayer:
    def __init__(self, modelId, device="auto"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(modelId)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelId, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )
        self.numLayers = len(self.model.model.layers)

    def _calculateMetrics(self, currentLogits, baselineProbs):
        logits32 = currentLogits.detach().to(torch.float32)
        probs32 = baselineProbs.detach().to(torch.float32)
        currentLogProbs = F.log_softmax(logits32, dim=-1)
        klDiv = F.kl_div(currentLogProbs, probs32, reduction='batchmean').item()
        softmaxProbs = F.softmax(logits32, dim=-1)
        entropy = -torch.sum(softmaxProbs * torch.log(softmaxProbs + 1e-9), dim=-1).item()
        return klDiv, entropy

    def runStructuralAnalysis(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            baselineOutput = self.model(**inputs)
            baselineLogits = baselineOutput.logits[:, -1, :]
            baselineProbs = F.softmax(baselineLogits.float(), dim=-1)
            hBase = -torch.sum(baselineProbs * torch.log(baselineProbs + 1e-9), dim=-1).item()

        levResults = []
        for i in tqdm(range(self.numLayers)):
            targetLayer = self.model.model.layers[i]
            layerBackup = copy.deepcopy(targetLayer)
            
            targetModules = {
                "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], 
                "mlp": ["gate_proj", "up_proj", "down_proj"]
            }

            for moduleType, projNames in targetModules.items():
                parent = targetLayer.self_attn if moduleType == "attn" else targetLayer.mlp
                for x in projNames:
                    oldProj = getattr(parent, x)
                    newProj = bitsandbytes.nn.modules.Linear4bit(
                        oldProj.in_features, 
                        oldProj.out_features, 
                        bias=(oldProj.bias is not None), 
                        compute_dtype=torch.float16
                    )
                    newProj.weight = bitsandbytes.nn.modules.Params4bit(
                        oldProj.weight.data.clone(), 
                        requires_grad=False,
                    )
                    if oldProj.bias is not None:
                        newProj.bias = torch.nn.Parameter(oldProj.bias.data.clone())
                    newProj.to(self.model.device)
                    setattr(parent, x, newProj)

            with torch.no_grad():
                currentOutput = self.model(**inputs)
                currentLogits = currentOutput.logits[:, -1, :]
                klVal, hVal = self._calculateMetrics(currentLogits, baselineProbs)
            
            levResults.append({
                "layer": i, 
                "kl": klVal, 
                "deltaH": abs(hVal - hBase)
            })

            self.model.model.layers[i] = layerBackup

        return levResults