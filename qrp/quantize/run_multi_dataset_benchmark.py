import argparse
import csv
import gc
import json
import math
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from qrp.model_mapper import get_layer_structure, get_model_layers


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
from qrp.external.awq_baseline import apply_awq_uniform
from qrp.external.atom_baseline import apply_atom_uniform
from qrp.external.gptq_baseline import (
    apply_gptq_uniform,
    build_calibration_batches,
    estimate_uniform_bits_size_bytes,
)
from qrp.external.slim_baseline import apply_slim_uniform
from qrp.external.smoothquant_baseline import apply_smoothquant_uniform
from qrp.external.spqr_baseline import apply_spqr_uniform, estimate_spqr_size_bytes
from qrp.quantize.quantizer import TargetedQuantizer


DATASET_REGISTRY = {
    "gsm8k": {
        "hf_path": ("gsm8k", "main"),
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",
        "display": "GSM8K",
    },
    "tfqa": {
        "hf_path": ("truthful_qa", "generation"),
        "split": "validation",
        "question_key": "question",
        "answer_key": "best_answer",
        "display": "TruthfulQA",
    },
    "mmlu": {
        "hf_path": ("cais/mmlu", "all"),
        "split": "test",
        "question_key": "question",
        "answer_key": "answer",   
        "display": "MMLU",
    },
}


def load_eval_pairs(dataset_key, n_samples, seed=0):
    cfg = DATASET_REGISTRY[dataset_key]
    ds = load_dataset(*cfg["hf_path"], split=cfg["split"])
    items = list(ds)
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n_samples]
    pairs = []
    for item in items:
        q = item[cfg["question_key"]]
        a = item[cfg["answer_key"]]
        if dataset_key == "mmlu":
            choices = item.get("choices", [])
            a = choices[int(a)] if choices else str(a)
        pairs.append((str(q), str(a)))
    return pairs


def evaluate_dataset(model, tokenizer, pairs, dataset_key):
    model.eval()
    total_loss = 0.0
    valid = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)

    for question, answer in tqdm(pairs, desc=f"  Evaluating {dataset_key}", leave=False):
        if dataset_key == "gsm8k":
            prompt = f"Question: {question}\nAnswer: Let's think step by step\n"
        else:
            prompt = f"Question: {question}\nAnswer: "

        prompt_ids = tokenizer.encode(prompt)
        target_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not target_ids:
            continue

        input_ids = torch.tensor([prompt_ids + target_ids]).to(model.device)
        labels = torch.tensor([[-100] * len(prompt_ids) + target_ids]).to(model.device)

        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss.item()

        if not math.isnan(loss) and not math.isinf(loss):
            total_loss += loss
            valid += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
        peak_bytes = torch.cuda.max_memory_allocated(model.device)
        peak_mb = peak_bytes / (1024 * 1024)
    else:
        peak_mb = 0.0

    if valid == 0:
        return 0.0, peak_mb
    return math.exp(-total_loss / valid), peak_mb


def estimate_size(model, layer_configs, num_layers):
    bytes_per_param = {"bf16": 2.0, "8bit": 1.0, "4bit": 0.5}
    total = 0.0
    layers = get_model_layers(model)
    for idx in range(num_layers):
        layer = layers[idx]
        (attn_parent, attn_projs), (mlp_parent, mlp_projs) = get_layer_structure(layer)
        
        params = 0
        if attn_parent is not None:
            for n in attn_projs:
                proj = getattr(attn_parent, n, None)
                if proj is not None and hasattr(proj, 'weight'):
                    params += proj.weight.numel()
        
        if mlp_parent is not None:
            for n in mlp_projs:
                proj = getattr(mlp_parent, n, None)
                if proj is not None and hasattr(proj, 'weight'):
                    params += proj.weight.numel()
                    
        total += params * bytes_per_param.get(layer_configs.get(idx, "bf16"), 2.0)
    return total


def write_csv(rows, dataset_displays, path):
    ds_cols   = [d for d in dataset_displays]
    eff_cols  = [f"Eff {d}" for d in dataset_displays]
    fieldnames = ["Model", "Size (MB)", "VRAM (MB)", "Compression"] + ds_cols + eff_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows, dataset_displays, model_name, path):
    n_ds = len(dataset_displays)
    col_spec = "l" + "r" * (3 + n_ds)
    ds_header = " & ".join(dataset_displays)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{Quantization Benchmark: Baselines vs. LLM-QRP ({model_name})}}",
        r"  \label{tab:benchmark}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    \\textbf{{Method}} & \\textbf{{Size (MB)}} & \\textbf{{VRAM (MB)}} "
            f"& \\textbf{{Compression}} & "
            + " & ".join(f"\\textbf{{{d}}}" for d in dataset_displays)
            + r" \\",
        r"    \midrule",
    ]

    ds_keys_list = list(dataset_displays)
    for row in rows:
        if "LLM-QRP" in row["Model"]:
            lines.append(r"    \midrule")
        size_str = f"{row['Size (MB)']:.1f}"
        vram_str = f"{row['VRAM (MB)']:.1f}" if row.get('VRAM (MB)', 0) > 0 else "-"
        cmpr_str = row["Compression"]
        ds_vals = []
        for d in ds_keys_list:
            val = row[d]
            ds_vals.append(f"{val:.4f}")
        ds_str = " & ".join(ds_vals)
        lines.append(
            f"    {row['Model']} & {size_str} & {vram_str} & {cmpr_str} & {ds_str} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset benchmark: BF16 / Uniform INT8 / Uniform INT4 baselines "
                    "vs optimal mixed-precision model."
    )
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples per dataset")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for evaluation sample selection")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to output JSON/CSV/TEX filenames (e.g. a seed)")
    parser.add_argument("--datasets", type=str, default="gsm8k,tfqa",
                        help="Comma-separated: gsm8k, tfqa, mmlu")
    parser.add_argument("--with-gptq", action="store_true",
                        help="Also benchmark uniform GPTQ as an external baseline")
    parser.add_argument("--gptq-bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Bit width for the GPTQ baseline (default: 4)")
    parser.add_argument("--gptq-samples", type=int, default=32,
                        help="Number of calibration sequences for GPTQ")
    parser.add_argument("--gptq-seqlen", type=int, default=2048,
                        help="Sequence length of each GPTQ calibration sequence")
    parser.add_argument("--gptq-percdamp", type=float, default=0.01,
                        help="GPTQ Hessian damping percentage")
    parser.add_argument("--with-awq", action="store_true",
                        help="Also benchmark uniform AWQ as an external baseline")
    parser.add_argument("--awq-bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Bit width for the AWQ baseline (default: 4)")
    parser.add_argument("--awq-group-size", type=int, default=128,
                        help="Quantization group size for the AWQ baseline")
    parser.add_argument("--awq-samples", type=int, default=32,
                        help="Number of calibration sequences for AWQ")
    parser.add_argument("--awq-seqlen", type=int, default=2048,
                        help="Sequence length of each AWQ calibration sequence")
    parser.add_argument("--with-spqr", action="store_true",
                        help="Also benchmark uniform SpQR as an external baseline")
    parser.add_argument("--spqr-bits", type=int, default=3, choices=[2, 3, 4, 8],
                        help="Bit width for the SpQR base quantization (default: 3)")
    parser.add_argument("--spqr-group-size", type=int, default=16,
                        help="Weight group size for the SpQR baseline (default: 16)")
    parser.add_argument("--spqr-qq-bits", type=int, default=3,
                        help="Bits for SpQR's double-quantized scale/zero stats (default: 3)")
    parser.add_argument("--spqr-qq-group-size", type=int, default=16,
                        help="Group size for double-quantized scale/zero stats (default: 16)")
    parser.add_argument("--spqr-outlier-threshold", type=float, default=0.2,
                        help="Relative leave-one-out threshold for SpQR fp16 outliers "
                             "(default: 0.2; pass 'inf' to disable outliers)")
    parser.add_argument("--spqr-permutation", type=str, default="act_order",
                        choices=["identity", "act_order", "spearman"],
                        help="Input-feature permutation order for SpQR (default: act_order)")
    parser.add_argument("--spqr-percdamp", type=float, default=1.0,
                        help="SpQR Hessian damping percentage (default: 1.0)")
    parser.add_argument("--spqr-samples", type=int, default=32,
                        help="Number of calibration sequences for SpQR")
    parser.add_argument("--spqr-seqlen", type=int, default=2048,
                        help="Sequence length of each SpQR calibration sequence")
    parser.add_argument("--with-slim", action="store_true",
                        help="Also benchmark uniform SliM-LLM as an external baseline")
    parser.add_argument("--slim-bits", type=int, default=2, choices=[2, 3, 4, 5, 6, 7, 8],
                        help="Average bit-width for the SliM-LLM baseline (default: 2)")
    parser.add_argument("--slim-group-size", type=int, default=128,
                        help="Salience-group size for the SliM-LLM baseline (default: 128)")
    parser.add_argument("--slim-percdamp", type=float, default=0.01,
                        help="SliM-LLM Hessian damping percentage")
    parser.add_argument("--slim-lambda-salience", type=float, default=1.0,
                        help="Weight of salient-weight error in SliM-LLM's quantizer fit")
    parser.add_argument("--slim-metric", type=str, default="mse",
                        help="SliM-LLM quantizer parameter search metric")
    parser.add_argument("--slim-samples", type=int, default=32,
                        help="Number of calibration sequences for SliM-LLM")
    parser.add_argument("--slim-seqlen", type=int, default=2048,
                        help="Sequence length of each SliM-LLM calibration sequence")
    parser.add_argument("--with-smoothquant", action="store_true",
                        help="Also benchmark SmoothQuant (activation-smoothed per-channel absmax)")
    parser.add_argument("--smoothquant-bits", type=int, default=8,
                        help="Weight bit-width for the SmoothQuant baseline (default: 8)")
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5,
                        help="Smoothing parameter for SmoothQuant (default: 0.5)")
    parser.add_argument("--smoothquant-samples", type=int, default=32,
                        help="Number of calibration sequences for SmoothQuant")
    parser.add_argument("--smoothquant-seqlen", type=int, default=2048,
                        help="Sequence length of each SmoothQuant calibration sequence")
    parser.add_argument("--with-atom", action="store_true",
                        help="Also benchmark Atom (group quantization) as an external baseline")
    parser.add_argument("--atom-bits", type=int, default=4, choices=[2, 3, 4, 8],
                        help="Weight bit-width for the Atom baseline (default: 4)")
    parser.add_argument("--atom-group-size", type=int, default=128,
                        help="Quantization group size for the Atom baseline (default: 128)")
    parser.add_argument("--atom-sym", action="store_true", default=True,
                        help="Use symmetric quantization for Atom (default: True)")
    parser.add_argument("--atom-no-sym", dest="atom_sym", action="store_false",
                        help="Use asymmetric quantization for Atom")
    parser.add_argument("--atom-clip-ratio", type=float, default=1.0,
                        help="Clip ratio for Atom weight quantization range")
    parser.add_argument("--atom-quant-type", type=str, default="int",
                        choices=["int", "fp"],
                        help="Atom quantization type: 'int' for INT or 'fp' for FP4")
    parser.add_argument("--atom-samples", type=int, default=32,
                        help="Number of calibration sequences for Atom")
    parser.add_argument("--atom-seqlen", type=int, default=2048,
                        help="Sequence length of each Atom calibration sequence")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_keys = [d.strip() for d in args.datasets.split(",")]
    for k in dataset_keys:
        if k not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset '{k}'. Choose from: {list(DATASET_REGISTRY)}")

    model_safe   = args.model_name.replace("/", "_")
    base_dir     = os.path.join(args.output_folder, model_safe)
    quantize_dir = os.path.join(base_dir, "quantize")
    opt_json     = os.path.join(quantize_dir, "optimal_mixed_precision.json")

    if not os.path.exists(opt_json):
        raise FileNotFoundError(
            f"Could not find {opt_json}.\n"
            "Run find_optimal_mixed_precision.py first."
        )

    with open(opt_json) as f:
        opt_data = json.load(f)

    optimal          = opt_data["optimal_config"]
    if "component_configs" in optimal:
        opt_layer_configs = None
        opt_comp_configs  = optimal["component_configs"]
        opt_bpw           = optimal.get("bits_per_param", 8.0)
        base_bytes        = opt_data.get("baseline_size_bytes")
    else:
        opt_layer_configs = {int(k): v for k, v in optimal["layer_configs"].items()}
        opt_comp_configs  = None
        opt_bpw           = None
        base_bytes        = None
    short_name  = args.model_name.split("/")[-1]   
    model_label_base = short_name                   
    model_label_opt  = f"+LLM-QRP (quantized)"

    print(f"\n{'='*65}")
    print(f"  MULTI-DATASET BENCHMARK")
    print(f"  Model:    {args.model_name}")
    print(f"  Datasets: {', '.join(dataset_keys)}")
    print(f"  Samples:  {args.samples} per dataset")
    if opt_comp_configs is not None:
        opt_desc = (f"{optimal.get('n_4bit', 0)} x FP4  |  "
                    f"{optimal.get('n_6bit', 0)} x 6bit  |  "
                    f"{optimal.get('n_8bit', 0)} x INT8  |  "
                    f"{optimal.get('n_16bit', 0)} x BF16")
    else:
        opt_desc = (f"{optimal['n_4bit']} x FP4  |  {optimal['n_8bit_only']} x INT8  |  "
                    f"{optimal.get('n_bf16', '?')} x BF16")
    print(f"  Optimal:  {opt_desc}")
    if args.with_gptq:
        print(f"  GPTQ:     uniform {args.gptq_bits}-bit external baseline")
    if args.with_awq:
        print(f"  AWQ:      uniform {args.awq_bits}-bit external baseline")
    if args.with_spqr:
        print(f"  SpQR:     uniform {args.spqr_bits}-bit sparse external baseline")
    if args.with_slim:
        print(f"  SliM-LLM: ~{args.slim_bits}-bit salience-mixed external baseline")
    if args.with_smoothquant:
        print(f"  SmoothQuant: {args.smoothquant_bits}-bit per-channel absmax (alpha={args.smoothquant_alpha})")
    if args.with_atom:
        print(f"  Atom: {args.atom_bits}-bit group quant (group_size={args.atom_group_size})")
    print(f"{'='*65}\n")

    print("Loading model (BF16)...")
    quantizer  = TargetedQuantizer(args.model_name)
    num_layers = quantizer.num_layers

    outlier_channels = {}
    if opt_comp_configs is not None:
        oc_candidates = [
            os.path.join(model_safe, "gsm8k", "outlier_channels.json"),
            os.path.join(base_dir, "gsm8k", "outlier_channels.json"),
        ]
        oc_path = next((p for p in oc_candidates if os.path.exists(p)), None)
        if oc_path:
            with open(oc_path) as f:
                outlier_channels = json.load(f).get("outlier_channels", {})
            print(f"Loaded {len(outlier_channels)} outlier-protected components from {oc_path}")

    baseline_size_mb = estimate_size(quantizer.model, {}, num_layers) / 1e6
    if opt_comp_configs is not None:
        base_mb = (base_bytes or (baseline_size_mb * 1e6)) / 1e6
        opt_size_mb = base_mb * (opt_bpw / 16.0)
    else:
        opt_size_mb = estimate_size(quantizer.model, opt_layer_configs, num_layers) / 1e6
    compression = baseline_size_mb / opt_size_mb
    size_reduction_pct = (1 - opt_size_mb / baseline_size_mb) * 100

    uniform_int8_configs = {i: "8bit" for i in range(num_layers)}
    uniform_int4_configs = {i: "4bit" for i in range(num_layers)}
    int8_size_mb = estimate_size(quantizer.model, uniform_int8_configs, num_layers) / 1e6
    int4_size_mb = estimate_size(quantizer.model, uniform_int4_configs, num_layers) / 1e6

    print(f"Baseline layer size:  {baseline_size_mb:.2f} MB")
    print(f"Uniform INT8 size:    {int8_size_mb:.2f} MB "
          f"({(1 - int8_size_mb / baseline_size_mb) * 100:.1f}% smaller, {baseline_size_mb / int8_size_mb:.2f}x)")
    print(f"Uniform INT4 size:    {int4_size_mb:.2f} MB "
          f"({(1 - int4_size_mb / baseline_size_mb) * 100:.1f}% smaller, {baseline_size_mb / int4_size_mb:.2f}x)")
    print(f"Optimal layer size:   {opt_size_mb:.2f} MB "
          f"({size_reduction_pct:.1f}% smaller, {compression:.2f}x)")

    ds_results = {}
    dataset_displays = [DATASET_REGISTRY[k]["display"] for k in dataset_keys]

    base_conditions = [
        ("bf16",  "BF16 baseline",            None, None),
        ("int8",  "Uniform INT8 baseline",    uniform_int8_configs, None),
        ("int4",  "Uniform INT4 baseline",    uniform_int4_configs, None),
        ("mixed", "Optimal mixed-precision",  opt_layer_configs, opt_comp_configs),
    ]

    external_baselines = []
    if args.with_gptq:
        gptq_key = f"gptq{args.gptq_bits}"
        gptq_size_mb = estimate_uniform_bits_size_bytes(
            quantizer.model, num_layers, args.gptq_bits) / 1e6
        print(f"GPTQ ({args.gptq_bits}-bit) size: "
              f"{gptq_size_mb:.2f} MB "
              f"({(1 - gptq_size_mb / baseline_size_mb) * 100:.1f}% smaller, "
              f"{baseline_size_mb / gptq_size_mb:.2f}x)")
        external_baselines.append({
            "key": gptq_key,
            "label": f"GPTQ ({args.gptq_bits}-bit)",
            "size_mb": gptq_size_mb,
            "samples": args.gptq_samples,
            "seqlen": args.gptq_seqlen,
            "banner": f"GPTQ BASELINE ({args.gptq_bits}-bit)",
            "detail": (f"GSM8K train | {args.gptq_samples} x "
                       f"{args.gptq_seqlen} tokens | percdamp={args.gptq_percdamp}"),
            "apply": lambda model, calib: apply_gptq_uniform(
                model, calib, wbits=args.gptq_bits, percdamp=args.gptq_percdamp),
        })
    if args.with_awq:
        awq_key = f"awq{args.awq_bits}"
        awq_size_mb = estimate_uniform_bits_size_bytes(
            quantizer.model, num_layers, args.awq_bits) / 1e6
        print(f"AWQ ({args.awq_bits}-bit) size: "
              f"{awq_size_mb:.2f} MB "
              f"({(1 - awq_size_mb / baseline_size_mb) * 100:.1f}% smaller, "
              f"{baseline_size_mb / awq_size_mb:.2f}x)")
        external_baselines.append({
            "key": awq_key,
            "label": f"AWQ ({args.awq_bits}-bit)",
            "size_mb": awq_size_mb,
            "samples": args.awq_samples,
            "seqlen": args.awq_seqlen,
            "banner": f"AWQ BASELINE ({args.awq_bits}-bit)",
            "detail": (f"GSM8K train | {args.awq_samples} x "
                       f"{args.awq_seqlen} tokens | group_size={args.awq_group_size}"),
            "apply": lambda model, calib: apply_awq_uniform(
                model, calib, wbits=args.awq_bits, q_group_size=args.awq_group_size),
        })
    if args.with_spqr:
        spqr_key = f"spqr{args.spqr_bits}"

        def _spqr_size_mb(outlier_share):
            return estimate_spqr_size_bytes(
                quantizer.model, num_layers,
                wbits=args.spqr_bits,
                groupsize=args.spqr_group_size,
                qq_scale_bits=args.spqr_qq_bits,
                qq_zero_bits=args.spqr_qq_bits,
                qq_groupsize=args.spqr_qq_group_size,
                outlier_share=outlier_share,
            ) / 1e6

        spqr_size_mb = _spqr_size_mb(0.0)
        print(f"SpQR ({args.spqr_bits}-bit) size: "
              f"{spqr_size_mb:.2f} MB "
              f"({(1 - spqr_size_mb / baseline_size_mb) * 100:.1f}% smaller, "
              f"{baseline_size_mb / spqr_size_mb:.2f}x)")
        external_baselines.append({
            "key": spqr_key,
            "label": f"SpQR ({args.spqr_bits}-bit)",
            "size_mb": spqr_size_mb,
            "samples": args.spqr_samples,
            "seqlen": args.spqr_seqlen,
            "banner": f"SpQR BASELINE ({args.spqr_bits}-bit)",
            "detail": (f"GSM8K train | {args.spqr_samples} x "
                       f"{args.spqr_seqlen} tokens | group_size={args.spqr_group_size} "
                       f"| outliers={'off' if math.isinf(args.spqr_outlier_threshold) else args.spqr_outlier_threshold}"),
            "finalize": lambda stats: _spqr_size_mb(stats["outlier_share"]),
            "apply": lambda model, calib: apply_spqr_uniform(
                model, calib,
                wbits=args.spqr_bits,
                groupsize=args.spqr_group_size,
                qq_scale_bits=args.spqr_qq_bits,
                qq_zero_bits=args.spqr_qq_bits,
                qq_groupsize=args.spqr_qq_group_size,
                outlier_threshold=args.spqr_outlier_threshold,
                permutation_order=args.spqr_permutation,
                percdamp=args.spqr_percdamp),
        })
    if args.with_slim:
        slim_key = f"slim{args.slim_bits}"
        # SliM-LLM promotes and demotes an equal number of weight groups
        # around `wbits`, so the average bit-width stays ~wbits; price it with
        # the same uniform-bit convention as the GPTQ/AWQ rows.
        slim_size_mb = estimate_uniform_bits_size_bytes(
            quantizer.model, num_layers, args.slim_bits) / 1e6
        print(f"SliM-LLM (~{args.slim_bits}-bit) size: "
              f"{slim_size_mb:.2f} MB "
              f"({(1 - slim_size_mb / baseline_size_mb) * 100:.1f}% smaller, "
              f"{baseline_size_mb / slim_size_mb:.2f}x)")
        external_baselines.append({
            "key": slim_key,
            "label": f"SliM-LLM ({args.slim_bits}-bit)",
            "size_mb": slim_size_mb,
            "samples": args.slim_samples,
            "seqlen": args.slim_seqlen,
            "banner": f"SliM-LLM BASELINE ({args.slim_bits}-bit)",
            "detail": (f"GSM8K train | {args.slim_samples} x "
                       f"{args.slim_seqlen} tokens | group_size={args.slim_group_size} "
                       f"| lambda_salience={args.slim_lambda_salience}"),
            "apply": lambda model, calib: apply_slim_uniform(
                model, calib,
                wbits=args.slim_bits,
                groupsize=args.slim_group_size,
                percdamp=args.slim_percdamp,
                metric=args.slim_metric,
                lambda_salience=args.slim_lambda_salience),
        })

    if args.with_smoothquant:
        sq_key = f"smoothquant{args.smoothquant_bits}"
        sq_size_mb = estimate_uniform_bits_size_bytes(
            quantizer.model, num_layers, args.smoothquant_bits) / 1e6
        print(f"SmoothQuant ({args.smoothquant_bits}-bit) size: "
              f"{sq_size_mb:.2f} MB "
              f"({(1 - sq_size_mb / baseline_size_mb) * 100:.1f}% smaller, "
              f"{baseline_size_mb / sq_size_mb:.2f}x)")
        external_baselines.append({
            "key": sq_key,
            "label": f"SmoothQuant ({args.smoothquant_bits}-bit)",
            "size_mb": sq_size_mb,
            "samples": args.smoothquant_samples,
            "seqlen": args.smoothquant_seqlen,
            "banner": f"SMOOTHQUANT BASELINE ({args.smoothquant_bits}-bit)",
            "detail": (f"GSM8K train | {args.smoothquant_samples} x "
                       f"{args.smoothquant_seqlen} tokens | alpha={args.smoothquant_alpha}"),
            "apply": lambda model, calib: apply_smoothquant_uniform(
                model, calib,
                wbits=args.smoothquant_bits,
                alpha=args.smoothquant_alpha),
        })

    if args.with_atom:
        atom_key = f"atom{args.atom_bits}"
        atom_size_mb = estimate_uniform_bits_size_bytes(
            quantizer.model, num_layers, args.atom_bits) / 1e6
        print(f"Atom ({args.atom_bits}-bit, group_size={args.atom_group_size}) size: "
              f"{atom_size_mb:.2f} MB "
              f"({(1 - atom_size_mb / baseline_size_mb) * 100:.1f}% smaller, "
              f"{baseline_size_mb / atom_size_mb:.2f}x)")
        external_baselines.append({
            "key": atom_key,
            "label": f"Atom ({args.atom_bits}-bit)",
            "size_mb": atom_size_mb,
            "samples": args.atom_samples,
            "seqlen": args.atom_seqlen,
            "banner": f"ATOM BASELINE ({args.atom_bits}-bit)",
            "detail": (f"GSM8K train | {args.atom_samples} x "
                       f"{args.atom_seqlen} tokens | group_size={args.atom_group_size} "
                       f"| sym={args.atom_sym}"),
            "apply": lambda model, calib: apply_atom_uniform(
                model, calib,
                wbits=args.atom_bits,
                weight_group_size=args.atom_group_size,
                w_sym=args.atom_sym,
                w_clip_ratio=args.atom_clip_ratio,
                quant_type=args.atom_quant_type),
        })

    method_sizes_mb = {
        "bf16":  baseline_size_mb,
        "int8":  int8_size_mb,
        "int4":  int4_size_mb,
        "mixed": opt_size_mb,
    }
    method_labels = {
        "bf16":  model_label_base,
        "int8":  "Uniform INT8",
        "int4":  "Uniform INT4",
        "mixed": model_label_opt,
    }
    for spec in external_baselines:
        method_sizes_mb[spec["key"]] = spec["size_mb"]
        method_labels[spec["key"]] = spec["label"]

    ds_pairs = {}
    ds_accs = {k: {} for k in dataset_keys}
    ds_vrams = {k: {} for k in dataset_keys}
    for ds_key in dataset_keys:
        display = DATASET_REGISTRY[ds_key]["display"]
        print(f"\n---- {display} ----")
        pairs = load_eval_pairs(ds_key, args.samples, seed=args.seed)
        ds_pairs[ds_key] = pairs

        for i, (key, desc, configs, comp_configs) in enumerate(base_conditions, 1):
            print(f"  [{i}/{len(base_conditions)}] {desc}...")
            if configs is None and comp_configs is None:
                quantizer.restore()
            elif comp_configs is not None:
                quantizer.quantize_components(comp_configs, outlier_channels=outlier_channels)
            else:
                quantizer.quantize_layers(configs)
            acc, vram = evaluate_dataset(
                quantizer.model, quantizer.tokenizer, pairs, ds_key)
            ds_accs[ds_key][key] = acc
            ds_vrams[ds_key][key] = vram
            print(f"  {desc}: {acc:.4f}")
    quantizer.restore()

    for spec in external_baselines:
        print(f"\n{'='*65}")
        print(f"  {spec['banner']}")
        print(f"  Calibration: {spec['detail']}")
        print(f"{'='*65}\n")
        calib_batches = build_calibration_batches(
            quantizer.tokenizer,
            nsamples=spec["samples"],
            seqlen=spec["seqlen"],
            device=next(quantizer.model.parameters()).device,
        )
        quantizer.restore()
        apply_stats = spec["apply"](quantizer.model, calib_batches)
        del calib_batches
        _empty_cache()
        if apply_stats is not None and "finalize" in spec:
            spec["measured"] = apply_stats
            spec["size_mb"] = spec["finalize"](apply_stats)
            method_sizes_mb[spec["key"]] = spec["size_mb"]
            print(f"  {spec['label']} measured: {apply_stats['effective_bpw']:.2f} avg bits/weight "
                  f"({apply_stats['outlier_share']:.2%} fp16 outliers) "
                  f"-> {spec['size_mb']:.2f} MB")
        for ds_key in dataset_keys:
            display = DATASET_REGISTRY[ds_key]["display"]
            print(f"  Evaluating {spec['label']} on {display}...")
            acc, vram = evaluate_dataset(
                quantizer.model, quantizer.tokenizer,
                ds_pairs[ds_key], ds_key)
            ds_accs[ds_key][spec["key"]] = acc
            ds_vrams[ds_key][spec["key"]] = vram
            print(f"  {spec['label']} [{display}]: "
                  f"{acc:.4f}")
        quantizer.restore()

    method_order = ["bf16", "int8", "int4"]
    method_order += [s["key"] for s in external_baselines]
    method_order.append("mixed")

    max_vram = {ds: max(ds_vrams[ds].get(k, 0.0) for k in method_order)
                for ds in dataset_keys}

    ds_results = {}
    for ds_key in dataset_keys:
        display = DATASET_REGISTRY[ds_key]["display"]
        accs = ds_accs[ds_key]
        base_acc = accs.get("bf16", 0.0)
        methods = {}
        for key in method_order:
            size_mb   = method_sizes_mb[key]
            acc       = accs[key]
            eff       = acc / size_mb if size_mb > 0 else 0.0
            drop_pct  = (1 - acc / base_acc) * 100 if base_acc > 0 and key != "bf16" else 0.0
            eff_gain  = (eff / (base_acc / baseline_size_mb) - 1) * 100 \
                if base_acc > 0 and baseline_size_mb > 0 and key != "bf16" else 0.0
            methods[key] = {
                "acc":          acc,
                "size_mb":      size_mb,
                "eff":          eff,
                "acc_drop_pct": drop_pct,
                "eff_gain_pct": eff_gain,
            }

        ds_results[ds_key] = {
            "display": display,
            "methods": methods,
        }

    table_rows = []
    for key in method_order:
        size_mb = method_sizes_mb[key]
        peak_vrams = {ds: ds_vrams[ds].get(key, 0.0) for ds in dataset_keys}
        max_vram_for_method = max(peak_vrams.values()) if peak_vrams else 0.0
        row = {
            "Model":       method_labels[key],
            "Size (MB)":   size_mb,
            "VRAM (MB)":   max_vram_for_method,
            "Compression": f"{baseline_size_mb / size_mb:.2f}x",
        }
        for ds_key, r in ds_results.items():
            d = r["display"]
            row[d] = r["methods"][key]["acc"]
            row[f"Eff {d}"] = r["methods"][key]["eff"]
        table_rows.append(row)

    print(f"\n{'='*75}")
    print("  RESULTS SUMMARY")
    print(f"{'='*75}")
    ds_col_w = 10
    header = f"  {'Model':<28} {'Size':>9} {'VRAM':>9} {'Cmpr':>7}" + "".join(
        f"  {r['display']:>{ds_col_w}}" for r in ds_results.values()
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in table_rows:
        ds_vals = "".join(
            f"  {row[DATASET_REGISTRY[k]['display']]:>{ds_col_w}.4f}"
            for k in dataset_keys
        )
        print(f"  {row['Model']:<28} {row['Size (MB)']:>8.1f}  {row['VRAM (MB)']:>8.1f}  {row['Compression']:>6}{ds_vals}")

    print(f"\n  Size:  {baseline_size_mb:.2f} MB -> {opt_size_mb:.2f} MB  "
          f"({size_reduction_pct:.1f}% reduction, {compression:.2f}x compression)")
    for ds_key, r in ds_results.items():
        print(f"  {r['display']:<12}")
        for key in method_order[1:]:
            m = r["methods"][key]
            drop_str = f"{m['acc_drop_pct']:.2f}%"
            gain_str = f"+{m['eff_gain_pct']:.1f}%" if m['eff_gain_pct'] >= 0 else f"{m['eff_gain_pct']:.1f}%"
            print(f"    {method_labels[key]:<22} Acc drop: {drop_str:>6}   Efficiency gain: {gain_str}")

    os.makedirs(quantize_dir, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    csv_path = os.path.join(quantize_dir, f"benchmark_results{suffix}.csv")
    tex_path = os.path.join(quantize_dir, f"benchmark_results{suffix}.tex")
    write_csv(table_rows, dataset_displays, csv_path)
    write_latex(table_rows, dataset_displays, args.model_name, tex_path)

    json_path = os.path.join(quantize_dir, f"multi_dataset_benchmark{suffix}.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_name":             args.model_name,
            "seed":                   args.seed,
            "samples_per_dataset":    args.samples,
            "baseline_size_mb":       baseline_size_mb,
            "uniform_int8_size_mb":   int8_size_mb,
            "uniform_int4_size_mb":   int4_size_mb,
            "optimal_size_mb":        opt_size_mb,
            "size_reduction_pct":     size_reduction_pct,
            "compression_ratio":      compression,
            "optimal_layer_config":   optimal,
            "external_baselines": [
                {
                    "name": s["label"],
                    "key": s["key"],
                    "size_mb": round(s["size_mb"], 4),
                    "calib_samples": s["samples"],
                    "calib_seqlen": s["seqlen"],
                    **({"percdamp": args.gptq_percdamp}
                       if s["key"].startswith("gptq")
                       else {"group_size": args.awq_group_size}
                       if s["key"].startswith("awq")
                       else {"group_size": args.spqr_group_size,
                             "qq_bits": args.spqr_qq_bits,
                             "qq_group_size": args.spqr_qq_group_size,
                             "outlier_threshold": args.spqr_outlier_threshold,
                             "permutation_order": args.spqr_permutation,
                             "percdamp": args.spqr_percdamp}
                       if s["key"].startswith("spqr")
                        else {"group_size": args.slim_group_size,
                              "lambda_salience": args.slim_lambda_salience,
                              "metric": args.slim_metric}
                        if s["key"].startswith("slim")
                         else {"alpha": args.smoothquant_alpha}
                         if s["key"].startswith("smoothquant")
                         else {"group_size": args.atom_group_size,
                               "sym": args.atom_sym,
                               "clip_ratio": args.atom_clip_ratio,
                               "quant_type": args.atom_quant_type}),
                    **({"measured_outlier_share": s["measured"]["outlier_share"],
                        "avg_bits_per_weight": s["measured"]["effective_bpw"]}
                       if "measured" in s else {}),
                }
                for s in external_baselines
            ],
            "dataset_results": {
                k: {
                    "display": v["display"],
                    "accuracy": {
                        m: round(v["methods"][m]["acc"], 6) for m in method_order
                    },
                    "size_mb": {
                        m: round(v["methods"][m]["size_mb"], 4) for m in method_order
                    },
                    "vram_mb": {
                        m: round(ds_vrams[k].get(m, 0.0), 2) for m in method_order
                    },
                    "efficiency": {
                        m: round(v["methods"][m]["eff"], 6) for m in method_order
                    },
                    "accuracy_drop_pct_vs_bf16": {
                        m: round(v["methods"][m]["acc_drop_pct"], 4) for m in method_order[1:]
                    },
                    "efficiency_gain_pct_vs_bf16": {
                        m: round(v["methods"][m]["eff_gain_pct"], 4) for m in method_order[1:]
                    },
                }
                for k, v in ds_results.items()
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()

