import argparse
import csv
import json
import math
import os

import torch
from datasets import load_dataset
from tqdm import tqdm

from qrp.model_mapper import get_layer_structure, get_model_layers
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
        f"  \\caption{{Quantization Benchmark: BF16 vs. LLM-QRP Optimal Mixed-Precision ({model_name})}}",
        r"  \label{tab:benchmark}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    \\textbf{{Model}} & \\textbf{{Size (MB)}} & \\textbf{{Compression}} & "
            + " & ".join(f"\\textbf{{{d}}}" for d in dataset_displays)
            + r" \\",
        r"    \midrule",
    ]

    for row in rows:
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
        description="Multi-dataset benchmark: BF16 vs optimal mixed-precision model."
    )
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of samples per dataset")
    parser.add_argument("--datasets", type=str, default="gsm8k,tfqa",
                        help="Comma-separated: gsm8k, tfqa, mmlu")
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
    print(f"{'='*65}\n")

    print("Loading model (BF16)...")
    quantizer  = TargetedQuantizer(args.model_name)
    num_layers = quantizer.num_layers
    baseline_size    = estimate_size(quantizer.model, {}, num_layers)
    baseline_size_mb = baseline_size / 1e6
    print(f"Baseline layer size:  {baseline_size_mb:.2f} MB")

    quantizer.quantize_layers(opt_layer_configs)
    opt_size    = estimate_size(quantizer.model, opt_layer_configs, num_layers)
    opt_size_mb = opt_size / 1e6
    compression = baseline_size_mb / opt_size_mb
    size_reduction_pct = (1 - opt_size_mb / baseline_size_mb) * 100
    print(f"Optimal layer size:   {opt_size_mb:.2f} MB  ({size_reduction_pct:.1f}% smaller, {compression:.2f}x)")
    quantizer.restore()

    ds_results = {}
    dataset_displays = [DATASET_REGISTRY[k]["display"] for k in dataset_keys]

    for ds_key in dataset_keys:
        display = DATASET_REGISTRY[ds_key]["display"]
        print(f"\n──── {display} ────")
        pairs = load_eval_pairs(ds_key, args.samples)

        print("  [1/2] BF16 baseline...")
        quantizer.restore()
        base_acc = evaluate_dataset(quantizer.model, quantizer.tokenizer, pairs, ds_key)
        print(f"  BF16:  {base_acc:.4f}")

        print("  [2/2] Optimal mixed-precision...")
        quantizer.quantize_layers(opt_layer_configs)
        opt_acc = evaluate_dataset(quantizer.model, quantizer.tokenizer, pairs, ds_key)
        print(f"  Mixed: {opt_acc:.4f}")
        quantizer.restore()

        acc_drop    = (1 - opt_acc / base_acc) * 100 if base_acc > 0 else 0.0
        base_eff    = base_acc / baseline_size_mb
        opt_eff     = opt_acc  / opt_size_mb
        eff_gain    = (opt_eff / base_eff - 1) * 100 if base_eff > 0 else 0.0

        ds_results[ds_key] = {
            "display":      display,
            "baseline_acc": base_acc,
            "opt_acc":      opt_acc,
            "acc_drop_pct": acc_drop,
            "baseline_eff": base_eff,
            "opt_eff":      opt_eff,
            "eff_gain_pct": eff_gain,
        }

    baseline_row = {
        "Model":         model_label_base,
        "Size (MB)":     baseline_size_mb,
        "Compression":   "1.00x",
    }
    opt_row = {
        "Model":         model_label_opt,
        "Size (MB)":     opt_size_mb,
        "Compression":   f"{compression:.2f}x",
    }
    for ds_key, r in ds_results.items():
        d = r["display"]
        baseline_row[d] = r["baseline_acc"]
        baseline_row[f"Eff {d}"] = r["baseline_eff"]
        opt_row[d]      = r["opt_acc"]
        opt_row[f"Eff {d}"] = r["opt_eff"]

    table_rows = [baseline_row, opt_row]

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
        drop_str = f"{r['acc_drop_pct']:.2f}%"
        gain_str = f"+{r['eff_gain_pct']:.1f}%" if r['eff_gain_pct'] >= 0 else f"{r['eff_gain_pct']:.1f}%"
        print(f"  {r['display']:<12} Acc drop: {drop_str:>6}   Efficiency gain: {gain_str}")

    os.makedirs(quantize_dir, exist_ok=True)
    csv_path = os.path.join(quantize_dir, "benchmark_results.csv")
    tex_path = os.path.join(quantize_dir, "benchmark_results.tex")
    write_csv(table_rows, dataset_displays, csv_path)
    write_latex(table_rows, dataset_displays, args.model_name, tex_path)

    json_path = os.path.join(quantize_dir, "multi_dataset_benchmark.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_name":           args.model_name,
            "samples_per_dataset":  args.samples,
            "baseline_size_mb":     baseline_size_mb,
            "optimal_size_mb":      opt_size_mb,
            "size_reduction_pct":   size_reduction_pct,
            "compression_ratio":    compression,
            "optimal_layer_config": optimal,
            "dataset_results": {
                k: {
                    "baseline_accuracy":  v["baseline_acc"],
                    "optimal_accuracy":   v["opt_acc"],
                    "accuracy_drop_pct":  v["acc_drop_pct"],
                    "baseline_efficiency": v["baseline_eff"],
                    "optimal_efficiency":  v["opt_eff"],
                    "efficiency_gain_pct": v["eff_gain_pct"],
                }
                for k, v in ds_results.items()
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()

