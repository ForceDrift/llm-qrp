import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
class AblationController:
    def __init__(self, modelId):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(modelId)
        self.model = AutoModelForCausalLM.from_pretrained(modelId, torch_dtype=torch.bfloat16).to(self.device)
        self.layerCount = (len(self.model.model.layers))

    def ablateLayer(self, layer):

        # self.model.layers[0].pop()
        layers = list(self.model.model.children())
        try:
            modules = layers[:layer] + layers[(layer+1):]
        except:
            modules = layers[:layer] + layers[len(layer)-1]
        model = nn.Sequential(*modules)        

        return model 

    

# to do: add benchmarking post-ablation for each layer pass, save to results
