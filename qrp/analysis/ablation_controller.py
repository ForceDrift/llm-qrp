import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

class AblationController:
    def __init__(self, modelId):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(modelId)
        self.model = AutoModelForCausalLM.from_pretrained(modelId, torch_dtype=torch.bfloat16).to(self.device)

        self.layerCount = (len(self.model.model.layers))
    def ablateLayer(self, layer):
        None
    def restoreLayer(self, layer):
        None
    def generateText(self, prompt):
        None
def runTest():
    None

if __name__ == "__main__":
    runTest()


# to do: add benchmarking post-ablation for each layer pass, save to results
