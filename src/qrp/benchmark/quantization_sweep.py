"""
quantization_sweep.py — Phase 2 empirical quantization threshold benchmark.

Algorithm:
  1. Load sorted layer scores from layer_avg_scores.json (produced by aggregate_scores.py).
  2. For each (threshold, bit_width) pair in a sweep grid:
       - Layers with score BELOW threshold → apply FP{bit_width} quantization via bitsandbytes.
       - Layers with score ABOVE threshold → kept at full precision (bfloat16).
  3. Evaluate on GSM8K, record accuracy.
  4. Output a JSON table + CSV for the paper.

The score threshold is interpreted in the normalized [0, 1] space produced by aggregate_scores.

Usage:
    python -m qrp.benchmark.quantization_sweep \\
        --scores-file results/layer_avg_scores.json \\
        --model-name HuggingFaceTB/SmolLM2-360M \\
        --dataset gsm8k \\
        --limit 200 \\
        --thresholds 0.1 0.2 0.3 0.4 0.5 \\
        --bit-widths 4 8 \\
        --output-folder results/sweep/
"""

import argparse
import copy
import csv
import json
import os
import re
import time
from typing import List, Tuple

import bitsandbytes
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from qrp.analysis.aggregate_scores import getSortedLayers


# ---------------------------------------------------------------------------
# Data loading & Answer Extraction
# ---------------------------------------------------------------------------

def load_dataset_samples(dataset_name: str, limit: int) -> list:
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        prompts = [{"question": x["question"], "answer": x["answer"]} for x in ds]
    elif dataset_name == "strqa":
        ds = load_dataset("tasksource/strategy-qa", split="train")
        prompts = [{"question": x["question"], "answer": x["answer"]} for x in ds]
    elif dataset_name == "tfqa":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        prompts = [{"question": x["question"], "answer": x["best_answer"]} for x in ds]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    return prompts[:limit]


def extract_answer(dataset_name, model_output, gt_answer=None):
    if dataset_name == "gsm8k":
        # GT extraction
        match = re.search(r"####\s*([\d,\.\-]+)", str(gt_answer)) if gt_answer else None
        gt = match.group(1).replace(",", "").strip() if match else str(gt_answer).strip()
        
        # Model extraction
        patterns = [r"the answer is\s*([\d,\.\-]+)", r"####\s*([\d,\.\-]+)", r"([\d,\.\-]+)\s*$"]
        pred = ""
        for p in patterns:
            m = re.findall(p, model_output, re.IGNORECASE)
            if m: pred = m[-1].replace(",", "").strip(); break
        if not pred:
            nums = re.findall(r"[\-]?\d[\d,\.]*", model_output)
            if nums: pred = nums[-1].replace(",", "").strip()
        
        try:
            correct = abs(float(pred) - float(gt)) < 1e-6
        except:
            correct = pred == gt
        return pred, gt, correct

    elif dataset_name == "strqa":
        pred = "yes" if "yes" in model_output.lower() else "no"
        gt = str(gt_answer).lower().strip()
        return pred, gt, pred == gt

    elif dataset_name == "tfqa":
        pred = model_output.strip()
        gt = str(gt_answer).strip()
        return pred, gt, gt.lower() in pred.lower()

    return model_output, gt_answer, False


def evaluate_model(model, tokenizer, samples, dataset_name, device="cuda", verbose=True):
    correct_count = 0
    results = []
    
    # Ensure pad token is set for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nEvaluating {len(samples)} samples for {dataset_name}...")

    for i, sample in enumerate(samples):
        question = sample["question"]
        gt_answer = sample["answer"]
        
        prompt = f"Question: {question}\nAnswer:"
        if dataset_name == "gsm8k":
            prompt = f"Solve the following math problem step by step. End your answer with '#### <number>'.\n\n{prompt}"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Explicit progress printing
        print(f"  [Sample {i+1}/{len(samples)}] Generating...", end="", flush=True)
        
        t_start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        t_gen = time.time() - t_start
        
        model_output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred, gt, is_correct = extract_answer(dataset_name, model_output, gt_answer)
        
        if is_correct:
            correct_count += 1
            status = "CORRECT"
        else:
            status = "WRONG"
            
        print(f" Done ({t_gen:.1f}s) -> {status}")
            
        results.append({
            "question": question,
            "gt": gt,
            "pred": pred,
            "correct": is_correct
        })

    accuracy = correct_count / len(samples) if samples else 0
    return accuracy, results


def load_sorted_layers(scores_file: str) -> list:
    """Load sorted (layer_idx, score) list from aggregate_scores output."""
    with open(scores_file, "r") as f:
        data = json.load(f)
    if "sortedAscending" in data:
        return [(int(e[0]), float(e[1])) for e in data["sortedAscending"]]
    elif "layerAvgScores" in data:
        return getSortedLayers(data["layerAvgScores"])
    return getSortedLayers(data)


# ---------------------------------------------------------------------------
# Per-layer quantization helpers
# ---------------------------------------------------------------------------

def _replace_linear_4bit(parent_module, attr_name: str, device) -> None:
    old = getattr(parent_module, attr_name)
    new = bitsandbytes.nn.Linear4bit(
        old.in_features,
        old.out_features,
        bias=(old.bias is not None),
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    )
    new.weight = bitsandbytes.nn.Params4bit(
        old.weight.data.clone(),
        requires_grad=False,
        quant_type="nf4",
    )
    if old.bias is not None:
        new.bias = torch.nn.Parameter(old.bias.data.clone())
    new.to(device)
    setattr(parent_module, attr_name, new)


def _replace_linear_8bit(parent_module, attr_name: str, device) -> None:
    old = getattr(parent_module, attr_name)
    new = bitsandbytes.nn.Linear8bitLt(
        old.in_features,
        old.out_features,
        bias=(old.bias is not None),
        has_fp16_weights=False,
    )
    new.weight = bitsandbytes.nn.Int8Params(
        old.weight.data.clone().to(torch.int8),
        requires_grad=False,
    )
    if old.bias is not None:
        new.bias = torch.nn.Parameter(old.bias.data.clone())
    new.to(device)
    setattr(parent_module, attr_name, new)


_ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
_MLP_PROJS  = ["gate_proj", "up_proj", "down_proj"]


def quantize_layer(layer, bit_width: int, device) -> None:
    replace_fn = _replace_linear_4bit if bit_width == 4 else _replace_linear_8bit

    for proj in _ATTN_PROJS:
        if hasattr(layer.self_attn, proj):
            replace_fn(layer.self_attn, proj, device)

    for proj in _MLP_PROJS:
        if hasattr(layer.mlp, proj):
            replace_fn(layer.mlp, proj, device)


def apply_threshold_quantization(
    model,
    sorted_layers: List[Tuple[int, float]],
    threshold: float,
    bit_width: int,
    device: str,
) -> Tuple[list, list]:
    quantized = []
    kept_fp = []

    for layer_idx, score in sorted_layers:
        if score < threshold:
            quantize_layer(model.model.layers[layer_idx], bit_width, device)
            quantized.append(layer_idx)
        else:
            kept_fp.append(layer_idx)

    return quantized, kept_fp


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    model_name: str,
    scores_file: str,
    dataset_name: str,
    limit: int,
    thresholds: List[float],
    bit_widths: List[int],
    output_folder: str,
    verbose: bool = False,
):
    print(f"\n{'='*65}")
    print(f"Quantization Sweep Benchmark")
    print(f"  Model:       {model_name}")
    print(f"  Dataset:     {dataset_name} (n={limit})")
    print(f"  Thresholds:  {thresholds}")
    print(f"  Bit-widths:  {bit_widths}")
    print(f"{'='*65}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Dtype detection
    if device == "cuda" and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print("Using BFLOAT16 for computation (Supported)")
    else:
        compute_dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Using {'FLOAT16' if device == 'cuda' else 'FLOAT32'} for computation")

    print("\n--- Phase 1: Loading Data ---")
    samples = load_dataset_samples(dataset_name, limit)
    sorted_layers = load_sorted_layers(scores_file)
    print(f"Loaded {len(samples)} samples, {len(sorted_layers)} scored layers.\n")

    print("--- Phase 2: Loading Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_run_results = []

    # 1. Baseline
    print(f"--- Running: baseline ({compute_dtype}, no quantization) ---")
    print(f"  Loading model: {model_name}...", end="", flush=True)
    t_load_start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=compute_dtype
    ).to(device)
    if device == "cuda": torch.cuda.synchronize()
    print(f" Done ({time.time() - t_load_start:.1f}s)")
    
    t0 = time.time()
    acc_base, _ = evaluate_model(base_model, tokenizer, samples, dataset_name, device=device, verbose=verbose)
    elapsed = time.time() - t0
    print(f"  Accuracy: {acc_base:.4f}  ({elapsed:.1f}s)\n")

    all_run_results.append({
        "threshold": None,
        "bit_width": None,
        "condition": "baseline",
        "num_layers_quantized": 0,
        "quantized_layer_indices": [],
        "accuracy": acc_base,
        "accuracy_drop": 0.0,
        "elapsed_sec": round(elapsed, 1),
    })

    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 2. Sweep
    for bit_width in bit_widths:
        for threshold in thresholds:
            run_label = f"threshold={threshold:.2f}, bit_width={bit_width}"
            print(f"--- Running: {run_label} ---")

            print(f"  Loading model: {model_name}...", end="", flush=True)
            t_load_start = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=compute_dtype
            ).to(device)
            if device == "cuda": torch.cuda.synchronize()
            print(f" Done ({time.time() - t_load_start:.1f}s)")

            print(f"  Applying selective quantization...", end="", flush=True)
            t_q_start = time.time()
            quantized_layers, kept_layers = apply_threshold_quantization(
                model, sorted_layers, threshold=threshold, bit_width=bit_width, device=device
            )
            if device == "cuda": torch.cuda.synchronize()
            print(f" Done ({time.time() - t_q_start:.2f}s)")
            
            print(f"  Quantized {len(quantized_layers)} layers: {sorted(quantized_layers)}")
            print(f"  Kept FP: {len(kept_layers)} layers")

            t0 = time.time()
            acc, _ = evaluate_model(model, tokenizer, samples, dataset_name, device=device, verbose=verbose)
            elapsed = time.time() - t0
            drop = acc_base - acc
            print(f"  Accuracy: {acc:.4f}  drop={drop:+.4f}  ({elapsed:.1f}s)\n")

            all_run_results.append({
                "threshold": threshold,
                "bit_width": bit_width,
                "condition": run_label,
                "num_layers_quantized": len(quantized_layers),
                "quantized_layer_indices": sorted(quantized_layers),
                "accuracy": acc,
                "accuracy_drop": drop,
                "elapsed_sec": round(elapsed, 1),
            })

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*65)
    print(f"{'Condition':<40} {'# Quantized':>12} {'Accuracy':>10} {'Drop':>8}")
    print("-"*65)
    for r in all_run_results:
        print(f"{r['condition']:<40} {r['num_layers_quantized']:>12} {r['accuracy']:>10.4f} {r['accuracy_drop']:>+8.4f}")
    print("="*65 + "\n")

    os.makedirs(output_folder, exist_ok=True)
    json_out = {
        "config": {
            "model_name": model_name,
            "dataset": dataset_name,
            "limit": limit,
            "thresholds": thresholds,
            "bit_widths": bit_widths,
            "scores_file": scores_file,
        },
        "results": all_run_results,
    }
    
    # Save results as dataset-specific files if possible, or generic
    json_file = os.path.join(output_folder, f"quantization_sweep_{dataset_name}.json")
    with open(json_file, "w") as f:
        json.dump(json_out, f, indent=2)
    
    csv_file = os.path.join(output_folder, f"quantization_sweep_{dataset_name}.csv")
    csv_fields = ["condition", "threshold", "bit_width", "num_layers_quantized", "accuracy", "accuracy_drop", "elapsed_sec"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_run_results)
        
    return json_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization Sweep Benchmark")
    parser.add_argument("--scores-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "strqa", "tfqa"], default="gsm8k")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--bit-widths", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run_sweep(
        model_name=args.model_name,
        scores_file=args.scores_file,
        dataset_name=args.dataset,
        limit=args.limit,
        thresholds=args.thresholds,
        bit_widths=args.bit_widths,
        output_folder=args.output_folder,
        verbose=args.verbose,
    )
