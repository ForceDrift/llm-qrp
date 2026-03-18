"""
Aggregate multi-model benchmark results into one combined LaTeX + CSV table.

Reads each model's quantize/multi_dataset_benchmark.json and produces:
  - combined_benchmark.csv   — one row per (model, variant) pair
  - combined_benchmark.tex   — LaTeX booktabs table with \multirow grouping
  - combined_benchmark.json  — raw merged data

Layout (matches paper style from image):

  Model                      | Size  | Cmpr  | GSM8K | TruthfulQA | ...
  ───────────────────────────────────────────────────────────────────────
  SmolLM2-135M      BF16     | 134.6 | 1.00x | 0.096 | 0.124      |
                    LLM-QRP  |  60.3 | 2.23x | 0.096 | 0.123      |
  ───────────────────────────────────────────────────────────────────────
  Granite-4.0-350M  BF16     | ...   | 1.00x | ...   | ...        |
                    LLM-QRP  | ...   | 2.xx x| ...   | ...        |
  ...
"""

import os
import csv
import json
import argparse

DATASET_REGISTRY = {
    "gsm8k": "GSM8K",
    "tfqa":  "TruthfulQA",
    "mmlu":  "MMLU",
}


def short_name(model_id: str) -> str:
    """Return a concise display name from a HuggingFace model ID."""
    part = model_id.split("/")[-1]
    # Normalise common verbose suffixes so table stays readable
    part = part.replace("-instruct", "").replace("-base", "").replace("-chat", "")
    return part


def load_model_results(model_id: str, output_folder: str) -> dict | None:
    """Load multi_dataset_benchmark.json for one model; return None if missing."""
    model_safe = model_id.replace("/", "_")
    path = os.path.join(output_folder, model_safe, "quantize", "multi_dataset_benchmark.json")
    if not os.path.exists(path):
        print(f"  WARNING: missing results for {model_id} at {path} — skipping")
        return None
    with open(path) as f:
        return json.load(f)


def write_csv(rows: list[dict], fieldnames: list[str], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV  → {path}")


def write_latex(model_blocks: list[dict], dataset_keys: list[str], out_path: str) -> None:
    """
    model_blocks: list of {
        "display": str,
        "baseline_size_mb": float,
        "optimal_size_mb": float,
        "compression_ratio": float,
        "datasets": { ds_key: {"baseline": float, "optimal": float, "drop": float} }
    }
    """
    ds_displays = [DATASET_REGISTRY.get(k, k.upper()) for k in dataset_keys]

    # Column spec: l (model) l (variant) r (size) r (compression) + one r per dataset
    n_ds = len(dataset_keys)
    col_spec = "ll" + "r" * (2 + n_ds)

    ds_header = " & ".join(f"\\textbf{{{d}}}" for d in ds_displays)

    lines = [
        r"\begin{table*}[h]",
        r"  \centering",
        r"  \caption{Benchmark Results: BF16 Baseline vs.\ LLM-QRP Optimal Mixed-Precision}",
        r"  \label{tab:multi_model_benchmark}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    \\textbf{{Model}} & \\textbf{{Precision}} & "
        r"\textbf{Size (MB)} & \textbf{Compression} & "
        + ds_header + r" \\",
        r"    \midrule",
    ]

    for i, block in enumerate(model_blocks):
        display   = block["display"]
        b_size    = block["baseline_size_mb"]
        o_size    = block["optimal_size_mb"]
        cmpr      = block["compression_ratio"]

        # BF16 row — model name spans both rows via \multirow
        bf16_ds = " & ".join(
            f"{block['datasets'][k]['baseline']:.4f}" for k in dataset_keys
        )
        lines.append(
            f"    \\multirow{{2}}{{*}}{{{display}}} & BF16 & "
            f"{b_size:.1f} & 1.00$\\times$ & {bf16_ds} \\\\"
        )

        # LLM-QRP row — model name left blank (covered by \multirow)
        opt_ds_parts = []
        for k in dataset_keys:
            base = block["datasets"][k]["baseline"]
            opt  = block["datasets"][k]["optimal"]
            drop = block["datasets"][k]["drop"]
            # Show value; if within 0.5% of baseline annotate with \approx
            if abs(drop) < 0.5:
                cell = f"\\textbf{{{opt:.4f}}}"
            else:
                delta = opt - base
                sign  = "+" if delta >= 0 else ""
                cell  = f"{opt:.4f} (\\textit{{{sign}{delta:.4f}}})"
            opt_ds_parts.append(cell)

        opt_ds = " & ".join(opt_ds_parts)
        lines.append(
            f"     & \\textbf{{LLM-QRP}} & "
            f"\\textbf{{{o_size:.1f}}} & \\textbf{{{cmpr:.2f}$\\times$}} & {opt_ds} \\\\"
        )

        # Add \midrule between model groups (not after the last)
        if i < len(model_blocks) - 1:
            lines.append(r"    \midrule")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved LaTeX → {out_path}")

    print()
    print("  ── LaTeX preview ────────────────────────────────────────")
    print("\n".join("  " + l for l in lines))
    print("  ─────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-model benchmark JSONs into a single combined table."
    )
    parser.add_argument("--output-folder", type=str, required=True,
                        help="Base folder used for all pipeline scripts")
    parser.add_argument("--models", type=str,
                        default="HuggingFaceTB/SmolLM2-135M,"
                                "ibm-granite/granite-4.0-350m-base,"
                                "Qwen/Qwen2.5-0.5B,"
                                "LiquidAI/LFM2-350M",
                        help="Comma-separated list of HuggingFace model IDs")
    parser.add_argument("--datasets", type=str, default="gsm8k,tfqa",
                        help="Comma-separated dataset keys matching what was evaluated")
    args = parser.parse_args()

    model_ids   = [m.strip() for m in args.models.split(",")]
    dataset_keys = [d.strip() for d in args.datasets.split(",")]

    print(f"\n{'='*65}")
    print(f"  AGGREGATING BENCHMARK RESULTS")
    print(f"  Models:   {', '.join(model_ids)}")
    print(f"  Datasets: {', '.join(dataset_keys)}")
    print(f"{'='*65}\n")

    model_blocks = []
    csv_rows     = []
    merged_json  = {}

    for model_id in model_ids:
        data = load_model_results(model_id, args.output_folder)
        if data is None:
            continue

        display       = short_name(model_id)
        b_size_mb     = data["baseline_size_mb"]
        o_size_mb     = data["optimal_size_mb"]
        cmpr          = data["compression_ratio"]
        size_red_pct  = data["size_reduction_pct"]
        ds_results    = data["dataset_results"]   # keyed by e.g. "gsm8k"

        # Build dataset sub-dict — handle missing keys gracefully
        ds_block = {}
        for k in dataset_keys:
            if k in ds_results:
                ds_block[k] = {
                    "baseline": ds_results[k]["baseline_accuracy"],
                    "optimal":  ds_results[k]["optimal_accuracy"],
                    "drop":     ds_results[k]["accuracy_drop_pct"],
                    "eff_gain": ds_results[k]["efficiency_gain_pct"],
                }
            else:
                ds_block[k] = {"baseline": float("nan"), "optimal": float("nan"),
                                "drop": float("nan"), "eff_gain": float("nan")}

        model_blocks.append({
            "model_id":         model_id,
            "display":          display,
            "baseline_size_mb": b_size_mb,
            "optimal_size_mb":  o_size_mb,
            "compression_ratio": cmpr,
            "size_reduction_pct": size_red_pct,
            "datasets":         ds_block,
        })

        # CSV — two rows per model (BF16 and LLM-QRP)
        base_row = {"Model": display, "Precision": "BF16",
                    "Size (MB)": round(b_size_mb, 2), "Compression": "1.00x"}
        opt_row  = {"Model": display, "Precision": "LLM-QRP",
                    "Size (MB)": round(o_size_mb, 2), "Compression": f"{cmpr:.2f}x"}

        for k in dataset_keys:
            col = DATASET_REGISTRY.get(k, k.upper())
            base_row[col] = round(ds_block[k]["baseline"], 4)
            opt_row[col]  = round(ds_block[k]["optimal"],  4)
            base_row[f"{col} Eff Gain"] = "—"
            opt_row[f"{col} Eff Gain"]  = f"+{ds_block[k]['eff_gain']:.1f}%"

        csv_rows.extend([base_row, opt_row])
        merged_json[model_id] = data

    if not model_blocks:
        print("ERROR: No model results found. Run the pipeline for each model first.")
        return

    # ── Print console summary ──────────────────────────────────────
    ds_w = 10
    print(f"\n{'='*75}")
    print("  COMBINED RESULTS")
    print(f"{'='*75}")
    ds_header_str = "".join(
        f"  {DATASET_REGISTRY.get(k, k.upper()):>{ds_w}}" for k in dataset_keys
    )
    print(f"  {'Model':<26} {'Prec':<10} {'MB':>7} {'Cmpr':>7}{ds_header_str}")
    print("  " + "-" * (55 + ds_w * len(dataset_keys)))

    for block in model_blocks:
        for variant, size_mb, cmpr_str in [
            ("BF16",    block["baseline_size_mb"], "1.00x"),
            ("LLM-QRP", block["optimal_size_mb"],  f"{block['compression_ratio']:.2f}x"),
        ]:
            model_label = block["display"] if variant == "BF16" else ""
            ds_vals = "".join(
                f"  {block['datasets'][k]['baseline' if variant == 'BF16' else 'optimal']:>{ds_w}.4f}"
                for k in dataset_keys
            )
            print(f"  {model_label:<26} {variant:<10} {size_mb:>7.1f} {cmpr_str:>7}{ds_vals}")
        print("  " + "-" * (55 + ds_w * len(dataset_keys)))

    # ── Save outputs ───────────────────────────────────────────────
    os.makedirs(args.output_folder, exist_ok=True)
    csv_path  = os.path.join(args.output_folder, "combined_benchmark.csv")
    tex_path  = os.path.join(args.output_folder, "combined_benchmark.tex")
    json_path = os.path.join(args.output_folder, "combined_benchmark.json")

    # CSV fieldnames — stable order
    ds_display_cols = [DATASET_REGISTRY.get(k, k.upper()) for k in dataset_keys]
    eff_cols        = [f"{d} Eff Gain" for d in ds_display_cols]
    fieldnames      = ["Model", "Precision", "Size (MB)", "Compression"] + ds_display_cols + eff_cols
    write_csv(csv_rows, fieldnames, csv_path)

    write_latex(model_blocks, dataset_keys, tex_path)

    with open(json_path, "w") as f:
        json.dump(merged_json, f, indent=2)
    print(f"  Saved JSON → {json_path}")

    print(f"\n  All combined outputs saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
