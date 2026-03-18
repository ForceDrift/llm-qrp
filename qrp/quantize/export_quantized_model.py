import os
import argparse
import json
import torch
from qrp.quantize.quantizer import TargetedQuantizer

def main():
    parser = argparse.ArgumentParser(description="Export a natively quantized mixed-precision model")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Base model name")
    parser.add_argument("--output-folder", type=str, required=True, help="Base folder where aggregated scores are saved")
    parser.add_argument("--threshold-4bit", type=float, required=True, help="Layers below this score will be quantized to 4-bit.")
    parser.add_argument("--threshold-8bit", type=float, required=True, help="Layers below this score but above the 4-bit threshold will be quantized to 8-bit.")
    
    args = parser.parse_args()
    
    model_name_safe = args.model_name.replace("/", "_")
    base_dir = os.path.join(args.output_folder, model_name_safe)
    aggregated_file = os.path.join(base_dir, "aggregated_scores.json")
    
    if not os.path.exists(aggregated_file):
        raise FileNotFoundError(f"Could not find {aggregated_file}. Please run run_analysis.py first.")
        
    with open(aggregated_file, "r") as f:
        scores = json.load(f)
        
    layer_configs = {}
    for layer_key, score in scores.items():
        layer_idx = int(layer_key.split("_")[1])
        if score < args.threshold_4bit:
            layer_configs[layer_idx] = "4bit"
        elif score < args.threshold_8bit:
            layer_configs[layer_idx] = "8bit"
            
    num_layers = len(scores)
    
    print(f"Loaded {num_layers} layers.")
    print(f"Quantizing {list(layer_configs.values()).count('4bit')} layers to 4-bit (Score < {args.threshold_4bit}).")
    print(f"Quantizing {list(layer_configs.values()).count('8bit')} layers to 8-bit (Score < {args.threshold_8bit}).")
    print(f"Layer distribution map: {layer_configs}")
    
    print("\nLoading Quantizer...")
    quantizer = TargetedQuantizer(args.model_name)
    
    print("\nApplying Quantization...")
    quantized_model = quantizer.quantize_layers(layer_configs)
    
    save_dir = os.path.join(base_dir, "saved_models", f"quantized_mixed_4b{args.threshold_4bit}_8b{args.threshold_8bit}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving mixed-precision model to {save_dir}...")
    torch.save(quantized_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    quantizer.model.config.save_pretrained(save_dir)
    quantizer.tokenizer.save_pretrained(save_dir)
    
    print("\nNote: Standard HF `from_pretrained` does not natively support loading piecemeal mixed-precision state mappings.")
    print("To reload this model in the future, you must instantiate the architecture, apply the same quantized layers via `quantizer.py`, and then `torch.load()` the state_dict!")
    print("Export Complete.")

if __name__ == "__main__":
    main()
