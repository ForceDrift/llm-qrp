import copy

import bitsandbytes
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

modelId = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(modelId)
tokenizer.pad_token = tokenizer.eos_token  # batch

model = AutoModelForCausalLM.from_pretrained(modelId, dtype=torch.bfloat16, device_map="auto")
# now load gsm8k
dataset = loadDataset("gsm8k", "main", split="test[:100]")  # make function
prompts = [f"Question: {i['question']} Think step by step \nAnswer:" for i in dataset]
# tokenize prompts
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

with torch.no_grad():
    baselineOutput = model(**inputs)
    # get out logits from the reasoning
    baselineLogits = baselineOutput.logits[:, -1, :]
    baselineProbs = F.softmax(baselineLogits.float(), dim=-1)

layerKL = []
for i in tqdm(range(len(model.model.layuers))):
    targetLayer = model.model.layers[i]
    originalAttnState = copy.deepcopy(targetLayer.self_attn.state_dict())  # store layers to restore after
