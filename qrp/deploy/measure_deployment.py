"""Deployment efficiency measurements: peak VRAM, latency, tokens/sec.

For a given model, reports per-method deployment figures:

  * ``peak_vram_mb``  - peak CUDA memory during a fixed-config generation run
                        (``torch.cuda.max_memory_allocated``), reset before each
                        method.
  * ``latency_s``     - wall-clock time to generate ``--output-tokens`` tokens
                        over ``--n-prompts`` prompts.
  * ``tokens_per_sec``- ``total_output_tokens / total_decode_seconds``.

Methods measured: BF16, Uniform INT8, Uniform INT4, and LLM-QRP (loaded from
``optimal_mixed_precision.json`` ``component_configs``).  External baseline
methods (GPTQ/AWQ/SpQR/...) are *not* reconstructed here; the script focuses on
the headline BF16 vs. uniform vs. QRP comparison used in Section 4.4.

Prompts are taken from GS-M8K (question only); generation uses greedy decoding
with a fixed ``max_new_tokens`` so the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from qrp.quantize.quantizer import TargetedQuantizer


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(n_prompts, seed=0):
    ds = load_dataset("gsm8k", "main", split="test")
    items = list(ds)
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n_prompts]
    return [f"Question: {it['question']}\nAnswer:" for it in items]


def measure(model, tokenizer, prompts, max_new_tokens, active_tokens):
    """Measure latency/tokens-sec/peak-VRAM over a controlled generation run."""
    if not torch.cuda.is_available():
        # CPU fallback: report latency only (VRAM = 0). Decoder loop is slow,
        # so clamp total tokens for speed.
        total_out = 0
        t0 = time.perf_counter()
        for p in prompts:
            tok = tokenizer.encode(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    tok, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                )
            total_out += out.shape[-1] - tok.shape[-1]
        elapsed = time.perf_counter() - t0
        return {
            "peak_vram_mb": 0.0,
            "latency_s": elapsed,
            "tokens_per_sec": total_out / elapsed if elapsed > 0 else 0.0,
            "total_output_tokens": total_out,
        }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)

    total_out = 0
    t0 = time.perf_counter()
    for p in tqdm(prompts, desc="  Generating", leave=False):
        tok = tokenizer.encode(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                tok,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        total_out += out.shape[-1] - tok.shape[-1]
    torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - t0
    peak_bytes = torch.cuda.max_memory_allocated(model.device)
    peak_mb = peak_bytes / (1024 * 1024)
    return {
        "peak_vram_mb": peak_mb,
        "latency_s": elapsed,
        "tokens_per_sec": total_out / elapsed if elapsed > 0 else 0.0,
        "total_output_tokens": total_out,
    }


def main():
    ap = argparse.ArgumentParser(description="Deployment efficiency measurement")
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--output-folder", type=str, required=True)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--output-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", type=str, default="bf16,int8,int4,mixed")
    args = ap.parse_args()

    set_seed(args.seed)
    methods = [m.strip() for m in args.methods.split(",")]

    model_safe = args.model_name.replace("/", "_")
    qdir = os.path.join(args.output_folder, model_safe, "quantize")
    out_dir = os.path.join(args.output_folder, "combined")
    os.makedirs(out_dir, exist_ok=True)

    quantizer = TargetedQuantizer(args.model_name)
    tokenizer = quantizer.tokenizer
    num_layers = quantizer.num_layers
    prompts = load_prompts(args.n_prompts, seed=args.seed)

    # Estimate sizes (MB) using the same convention as the benchmark.
    def _bf16_mb():
        total = sum(
            p.numel() for p in quantizer.model.parameters()
        ) * 2.0 / 1e6
        return total

    size_mb = {}
    uniform_int8 = {i: "8bit" for i in range(num_layers)}
    uniform_int4 = {i: "4bit" for i in range(num_layers)}

    mixed_comp = None
    mixed_bpw = None
    base_bytes = None
    opt_json = os.path.join(qdir, "optimal_mixed_precision.json")
    if os.path.exists(opt_json):
        with open(opt_json) as f:
            od = json.load(f)
        opt = od["optimal_config"]
        if "component_configs" in opt:
            mixed_comp = opt["component_configs"]
            mixed_bpw = opt.get("bits_per_param", 8.0)
            base_bytes = od.get("baseline_size_bytes")

    results = {}
    for method in methods:
        print(f"\n===== {method} =====")
        if method == "bf16":
            quantizer.restore()
            size_mb[method] = _bf16_mb()
        elif method == "int8":
            quantizer.restore()
            quantizer.quantize_layers(uniform_int8)
            size_mb[method] = _bf16_mb() / 2.0
        elif method == "int4":
            quantizer.restore()
            quantizer.quantize_layers(uniform_int4)
            size_mb[method] = _bf16_mb() / 4.0
        elif method == "mixed":
            quantizer.restore()
            if mixed_comp is None:
                print("[warn] no optimal config; skipping mixed")
                continue
            quantizer.quantize_components(mixed_comp)
            size_mb[method] = (base_bytes or (_bf16_mb() * 1e6)) * (mixed_bpw / 16.0) / 1e6
        else:
            print(f"[warn] unknown method '{method}', skipping")
            continue

        meas = measure(quantizer.model, tokenizer, prompts,
                       args.output_tokens, args.output_tokens)
        meas["size_mb"] = size_mb[method]
        results[method] = meas
        print(f"  size={meas['size_mb']:.1f} MB  "
              f"peak_vram={meas['peak_vram_mb']:.1f} MB  "
              f"latency={meas['latency_s']:.3f}s  "
              f"tok/s={meas['tokens_per_sec']:.1f}")

    quantizer.restore()

    json_path = os.path.join(out_dir, f"deployment_{model_safe}.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "n_prompts": args.n_prompts,
            "output_tokens": args.output_tokens,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
