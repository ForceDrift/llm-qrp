import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import bitsandbytes
import copy



modelId = "Qwen/Qwen2.5-1.5B" 
tokenizer = AutoTokenizer.from_pretrained(modelId)
model1 = AutoModelForCausalLM.from_pretrained(
    modelId, 
    dtype=torch.bfloat16, #intital 16 bit precision but we change later
    device_map="auto"
)
prompt = "If a plane crashes on the border of the US and Canada, where do you bury the survivors?"
inputs = tokenizer(prompt, return_tensors="pt").to(model1.device)
with torch.no_grad():
    baselineOutput = model1(**inputs)
    baselineLogits = baselineOutput.logits[:, -1, :] #get all logits from the prompt running

def calculateEntropy(logits):
    prob=torch.softmax(logits.float(), dim=-1)
    return -torch.sum(prob * torch.log(prob + 1e-9), dim=-1).item()

hBase = calculateEntropy(baselineLogits)
print(hBase)

levResults = []
print("-- layer analysis --")

for i in tqdm(range(len(model1.model.layers))):
    targetLayer = model1.model.layers[i]
    originalAttnState = copy.deepcopy(targetLayer.self_attn.state_dict()) #copy the layer weights to be restored after changing after
    
    for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']: #for each matrice (query, key, value, output) compress
        oldProj = getattr(targetLayer.self_attn, x)
        newProj = bitsandbytes.nn.modules.Linear4bit(
            oldProj.in_features, 
            oldProj.out_features, 
            bias=(oldProj.bias is not None),
            compute_dtype=torch.float16
        )
        newWeight = bitsandbytes.nn.modules.Params4bit(
            oldProj.weight.data.clone(),
            requires_grad=False,
        )
        
        newProj.weight = newWeight
        if oldProj.bias is not None:
            newProj.bias = torch.nn.Parameter(oldProj.bias.data.clone())
            
        newProj.to(model1.device)
        setattr(targetLayer.self_attn, x, newProj)

    with torch.no_grad():
        currentOutput = model1(**inputs)
        currentLogits = currentOutput.logits[:, -1, :]
        hLayer = calculateEntropy(currentLogits)
    
    deltaH = abs(hLayer - hBase)
    levResults.append(deltaH)

    for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
 
        backupWeight = originalAttnState[f"{x}.weight"]
        hasBias = f"{x}.bias" in originalAttnState
        
        resetProj = torch.nn.Linear(
            backupWeight.shape[1], 
            backupWeight.shape[0], 
            bias=hasBias
        ).to(model1.dtype).to(model1.device)
        
        setattr(targetLayer.self_attn, x, resetProj)
    
    targetLayer.self_attn.load_state_dict(originalAttnState)

print("\n-- final results --")
for index, value in enumerate(levResults):
    print(f"layer{index}: {value:.6f}")