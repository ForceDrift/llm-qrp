import argparse
import csv
import gc
import json
import math
import os

import torch
from datasets import load_dataset
from tqdm import tqdm

from qrp.model_mapper import get_layer_structure, get_model_layers


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def load_eval_pairs(dataset_key, n_samples):
    cfg = DATASET_REGISTRY[dataset_key]
    ds = load_dataset(*cfg["hf_path"], split=cfg["split"])
    pairs = []
    for item in list(ds)[:n_samples]:
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

    if valid == 0:
        return 0.0
    return math.exp(-total_loss / valid)


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
    fieldnames = ["Model", "Size (MB)", "Compression"] + ds_cols + eff_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows, dataset_displays, model_name, path):
    n_ds = len(dataset_displays)
    col_spec = "l" + "r" * (2 + n_ds)   
    ds_header  = " & ".join(dataset_displays)
    eff_header = " & ".join(f"Eff ({d})" for d in dataset_displays)

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        f"  \\caption{{Quantization Benchmark: "
        f"{', '.join(r['Model'] for r in rows if 'LLM-QRP' not in r['Model'])} "
        f"vs. LLM-QRP Optimal Mixed-Precision ({model_name})}}",
        r"  \label{tab:benchmark}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    \\textbf{{Model}} & \\textbf{{Size (MB)}} & \\textbf{{Compression}} & "
            + " & ".join(f"\\textbf{{{d}}}" for d in dataset_displays)
            + r" \\",
        r"    \midrule",
    ]

    for row in rows:
        if "LLM-QRP" in row["Model"]:
            lines.append(r"    \midrule")
        size_str  = f"{row['Size (MB)']:.1f}"
        cmpr_str  = row["Compression"]
        ds_vals   = " & ".join(f"{row[d]:.4f}" for d in dataset_displays)
        lines.append(
            f"    {row['Model']} & {size_str} & {cmpr_str} & {ds_vals} \\\\"
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
    args = parser.parse_args()

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
    opt_layer_configs = {int(k): v for k, v in optimal["layer_configs"].items()}
    short_name  = args.model_name.split("/")[-1]   
    model_label_base = short_name                   
    model_label_opt  = f"+LLM-QRP (quantized)"

    print(f"\n{'='*65}")
    print(f"  MULTI-DATASET BENCHMARK")
    print(f"  Model:    {args.model_name}")
    print(f"  Datasets: {', '.join(dataset_keys)}")
    print(f"  Samples:  {args.samples} per dataset")
    opt_desc = (f"{optimal['n_4bit']} × FP4  |  {optimal['n_8bit_only']} × INT8  |  "
                f"{optimal.get('n_bf16', '?')} × BF16")
    print(f"  Optimal:  {opt_desc}")
    if args.with_gptq:
        print(f"  GPTQ:     uniform {args.gptq_bits}-bit external baseline")
    if args.with_awq:
        print(f"  AWQ:      uniform {args.awq_bits}-bit external baseline")
    print(f"{'='*65}\n")

    print("Loading model (BF16)...")
    quantizer  = TargetedQuantizer(args.model_name)
    num_layers = quantizer.num_layers

    baseline_size_mb = estimate_size(quantizer.model, {}, num_layers) / 1e6
    opt_size_mb      = estimate_size(quantizer.model, opt_layer_configs, num_layers) / 1e6
    compression      = baseline_size_mb / opt_size_mb
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
        ("bf16",  "BF16 baseline",            None),
        ("int8",  "Uniform INT8 baseline",    uniform_int8_configs),
        ("int4",  "Uniform INT4 baseline",    uniform_int4_configs),
        ("mixed", "Optimal mixed-precision",  opt_layer_configs),
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
    for ds_key in dataset_keys:
        display = DATASET_REGISTRY[ds_key]["display"]
        print(f"\n──── {display} ────")
        pairs = load_eval_pairs(ds_key, args.samples)
        ds_pairs[ds_key] = pairs

        for i, (key, desc, configs) in enumerate(base_conditions, 1):
            print(f"  [{i}/{len(base_conditions)}] {desc}...")
            if configs is None:
                quantizer.restore()
            else:
                quantizer.quantize_layers(configs)
            ds_accs[ds_key][key] = evaluate_dataset(
                quantizer.model, quantizer.tokenizer, pairs, ds_key)
            print(f"  {desc}: {ds_accs[ds_key][key]:.4f}")
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
        spec["apply"](quantizer.model, calib_batches)
        del calib_batches
        _empty_cache()
        for ds_key in dataset_keys:
            display = DATASET_REGISTRY[ds_key]["display"]
            print(f"  Evaluating {spec['label']} on {display}...")
            ds_accs[ds_key][spec["key"]] = evaluate_dataset(
                quantizer.model, quantizer.tokenizer,
                ds_pairs[ds_key], ds_key)
            print(f"  {spec['label']} [{display}]: "
                  f"{ds_accs[ds_key][spec['key']]:.4f}")
        quantizer.restore()

    method_order = ["bf16", "int8", "int4"]
    method_order += [s["key"] for s in external_baselines]
    method_order.append("mixed")

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
        row = {
            "Model":       method_labels[key],
            "Size (MB)":   size_mb,
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
    header = f"  {'Model':<28} {'Size':>9} {'Cmpr':>7}" + "".join(
        f"  {r['display']:>{ds_col_w}}" for r in ds_results.values()
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in table_rows:
        ds_vals = "".join(
            f"  {row[DATASET_REGISTRY[k]['display']]:>{ds_col_w}.4f}"
            for k in dataset_keys
        )
        print(f"  {row['Model']:<28} {row['Size (MB)']:>8.1f}  {row['Compression']:>6}{ds_vals}")

    print(f"\n  Size:  {baseline_size_mb:.2f} MB → {opt_size_mb:.2f} MB  "
          f"({size_reduction_pct:.1f}% reduction, {compression:.2f}x compression)")
    for ds_key, r in ds_results.items():
        print(f"  {r['display']:<12}")
        for key in method_order[1:]:
            m = r["methods"][key]
            drop_str = f"{m['acc_drop_pct']:.2f}%"
            gain_str = f"+{m['eff_gain_pct']:.1f}%" if m['eff_gain_pct'] >= 0 else f"{m['eff_gain_pct']:.1f}%"
            print(f"    {method_labels[key]:<22} Acc drop: {drop_str:>6}   Efficiency gain: {gain_str}")

    os.makedirs(quantize_dir, exist_ok=True)
    csv_path = os.path.join(quantize_dir, "benchmark_results.csv")
    tex_path = os.path.join(quantize_dir, "benchmark_results.tex")
    write_csv(table_rows, dataset_displays, csv_path)
    write_latex(table_rows, dataset_displays, args.model_name, tex_path)

    json_path = os.path.join(quantize_dir, "multi_dataset_benchmark.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_name":             args.model_name,
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
                       else {"group_size": args.awq_group_size}),
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

