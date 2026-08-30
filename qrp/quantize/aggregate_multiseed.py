"""Aggregate the 5-seed benchmark runs into mean +/- std (paper Section 4).

Each model has one ``multi_dataset_benchmark.json`` (the seed-0 run) plus
``multi_dataset_benchmark_seed{1..4}.json``.  This script reads all available
seeds for every model and, per (method, dataset), computes the mean and sample
standard deviation of the acceleration metric ``exp(-NLL)``.

Writes into ``results/combined``:

  * ``benchmark_multiseed.json`` - rich summary (per model/method/dataset:
    ``mean``, ``std``, ``min``, ``max``, ``n_seeds``).
  * ``benchmark_multiseed.csv`` - rows ``Model, Method, B_avg, Footprint(MB),
    Dataset, mean +/- std``.
  * ``benchmark_multiseed.tex`` - one iso-bit-style table per model with
    ``mu +/- sigma`` in every accuracy cell.

The metric, sizes, and methods match ``aggregate_benchmarks.py`` so the tables
are drop-in replacements for the single-run ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics

METHOD_ORDER = [
    "bf16", "int8", "int4",
    "gptq4", "awq4", "spqr3", "slim2", "smoothquant8", "atom4",
    "mixed",
]

METHOD_DISPLAY = {
    "bf16": "BF16",
    "int8": "Uniform INT8",
    "int4": "Uniform INT4",
    "gptq4": "GPTQ (4-bit)",
    "awq4": "AWQ (4-bit)",
    "spqr3": "SpQR (3-bit)",
    "slim2": "SliM-LLM (2-bit)",
    "smoothquant8": "SmoothQuant (8-bit)",
    "atom4": "Atom (4-bit)",
    "mixed": "LLM-QRP",
}

# Average bit-width per method (consistent with the single-run tables).
AVG_BITS = {
    "bf16": 16.0, "int8": 8.0, "int4": 4.0,
    "gptq4": 4.0, "awq4": 4.0, "spqr3": 3.0, "slim2": 2.0,
    "smoothquant8": 8.0, "atom4": 4.0, "mixed": None,  # set per model below
}

DATASET_ORDER = ["gsm8k", "tfqa", "mmlu"]
DATASET_DISPLAY = {"gsm8k": "GSM8K", "tfqa": "TruthfulQA", "mmlu": "MMLU"}


def collect_seed_files(results_dir, model_dir):
    base = os.path.join(results_dir, model_dir, "quantize", "multi_dataset_benchmark.json")
    files = []
    if os.path.isfile(base):
        files.append(base)
    for n in range(1, 8):
        p = os.path.join(results_dir, model_dir, "quantize",
                         f"multi_dataset_benchmark_seed{n}.json")
        if os.path.isfile(p):
            files.append(p)
    return files


def aggregate_model(results_dir, model_dir):
    files = collect_seed_files(results_dir, model_dir)
    if not files:
        return None

    loaded = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        loaded.append(d)

    model_name = loaded[0]["model_name"]
    datasets = [k for k in DATASET_ORDER if k in loaded[0]["dataset_results"]]
    # Per-method footprint (MB) - take from the seed-0 file for consistency.
    size_mb = {}
    for method in METHOD_ORDER:
        if method == "mixed":
            size_mb[method] = loaded[0]["dataset_results"][datasets[0]].get(
                "size_mb", {}).get("mixed", None)
        else:
            size_mb[method] = loaded[0]["dataset_results"][datasets[0]].get(
                "size_mb", {}).get(method, None)

    base_mb = loaded[0]["baseline_size_mb"]

    # Mixed bits/param derived from the size ratio (16-bit baseline -> mixed).
    mixed_mb = size_mb.get("mixed")
    if mixed_mb:
        mixed_bpw = 16.0 * mixed_mb / base_mb if base_mb > 0 else None
    else:
        mixed_bpw = None

    out = {"model_name": model_name, "model_dir": model_dir,
           "mixed_bpw": mixed_bpw, "n_seeds": len(loaded), "datasets": {}}
    for ds in datasets:
        acc_sets = {m: [] for m in METHOD_ORDER}
        for d in loaded:
            accs = d["dataset_results"][ds]["accuracy"]
            for m in METHOD_ORDER:
                if m in accs:
                    acc_sets[m].append(accs[m])
        out["datasets"][ds] = {}
        for m in METHOD_ORDER:
            vals = acc_sets[m]
            if not vals:
                continue
            out["datasets"][ds][m] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    return {"model": out, "size_mb": size_mb, "base_mb": base_mb}


def build_summary(results_dir):
    summary = {}
    for model_dir in sorted(os.listdir(results_dir)):
        if not os.path.isdir(os.path.join(results_dir, model_dir)):
            continue
        agg = aggregate_model(results_dir, model_dir)
        if agg is not None:
            summary[model_dir] = agg
    return summary


def fmt_ms(m):
    if m is None:
        return "-"
    return f"{m:.4f}"


def write_csv(summary, path):
    rows = []
    for model_dir, agg in summary.items():
        model = agg["model"]
        mixed_bpw = model["mixed_bpw"]
        for ds in DATASET_ORDER:
            if ds not in model["datasets"]:
                continue
            for m in METHOD_ORDER:
                if m not in model["datasets"][ds]:
                    continue
                mms = model["datasets"][ds][m]
                size = agg["size_mb"].get(m)
                if m == "mixed":
                    b_avg = f"{mixed_bpw:.2f}" if mixed_bpw else "-"
                else:
                    b_avg = f"{AVG_BITS[m]:.1f}"
                rows.append({
                    "Model": model["model_name"].split("/")[-1],
                    "Method": METHOD_DISPLAY[m],
                    "B_avg(bits)": b_avg,
                    "Footprint(MB)": fmt_ms(size),
                    "Dataset": DATASET_DISPLAY[ds],
                    "mean": round(mms["mean"], 4),
                    "std": round(mms["std"], 4),
                    "min": round(mms["min"], 4),
                    "max": round(mms["max"], 4),
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_tex(summary, path):
    lines = [
        r"\begin{table*}[htbp]",
        r"  \centering",
        r"  \small",
        r"\caption{Quantization benchmarks (mean $\pm$ std over 5 seeds) across all models. "
        r"The metric is the target-probability $\exp(-\text{NLL})$; best non-BF16 value per "
        r"dataset column is \textbf{bolded}.}",
        r"  \label{tab:multiseed}",
        r"  \begin{tabular}{llrrrrr}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Method} & $B_{\mathrm{avg}}$ (bits) "
        r"& \textbf{Footprint (MB)} & \textbf{GSM8K} & \textbf{TruthfulQA} & \textbf{MMLU} \\",
        r"    \midrule",
    ]

    first_model = True
    for model_dir, agg in summary.items():
        model = agg["model"]
        short = model["model_name"].split("/")[-1]
        mixed_bpw = model["mixed_bpw"]

        # Best non-BF16 mean per dataset for bolding.
        best = {}
        for ds in DATASET_ORDER:
            if ds not in model["datasets"]:
                continue
            best_mean = -1.0
            for m in METHOD_ORDER:
                if m == "bf16" or m not in model["datasets"][ds]:
                    continue
                best_mean = max(best_mean, model["datasets"][ds][m]["mean"])
            best[ds] = best_mean

        if not first_model:
            lines.append(r"    \midrule")
        first_model = False

        first_row = True
        for m in METHOD_ORDER:
            present = any(m in model["datasets"][ds] for ds in DATASET_ORDER)
            if not present:
                continue
            if m == "mixed":
                b_avg = f"{mixed_bpw:.2f}" if mixed_bpw else "-"
                size = agg["size_mb"].get(m)
                size_s = f"{size:.1f}" if size else "-"
                display = f"\\textbf{{{METHOD_DISPLAY[m]}}}"
            else:
                b_avg = f"{AVG_BITS[m]:.1f}"
                size = agg["size_mb"].get(m)
                size_s = f"{size:.1f}" if size else "-"
                display = METHOD_DISPLAY[m]

            model_col = short if first_row else ""
            first_row = False

            vals = []
            for ds in DATASET_ORDER:
                if ds not in model["datasets"] or m not in model["datasets"][ds]:
                    vals.append("-")
                    continue
                mms = model["datasets"][ds][m]
                txt = f"${mms['mean']:.4f} \\pm {mms['std']:.4f}$"
                if m != "bf16" and abs(mms["mean"] - best[ds]) < 1e-9:
                    txt = f"\\textbf{{{txt}}}"
                vals.append(txt)
            vals_s = " & ".join(vals)
            lines.append(f"    {model_col} & {display} & {b_avg} & {size_s} & {vals_s} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Aggregate 5-seed benchmarks into mu+/-std")
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    results_dir = args.results_dir
    out_dir = args.output_dir or os.path.join(results_dir, "combined")
    os.makedirs(out_dir, exist_ok=True)

    summary = build_summary(results_dir)
    if not summary:
        print("No benchmark results found.")
        return 1

    jp = os.path.join(out_dir, "benchmark_multiseed.json")
    cp = os.path.join(out_dir, "benchmark_multiseed.csv")
    tp = os.path.join(out_dir, "benchmark_multiseed.tex")

    # JSON (strip numpy-free plain floats).
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(summary, cp)
    write_tex(summary, tp)

    print(f"Wrote {jp}")
    print(f"Wrote {cp}")
    print(f"Wrote {tp}")
    for model_dir, agg in summary.items():
        model = agg["model"]
        print(f"  {model['model_name']}: n_seeds={model['n_seeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
