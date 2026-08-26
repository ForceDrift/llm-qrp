import argparse
import csv
import json
import os
import sys


METHOD_ORDER = [
    "bf16", "int8", "int4",
    "gptq4", "awq4", "spqr3", "slim2", "smoothquant8", "atom4",
    "mixed",
]

METHOD_DISPLAY = {
    "bf16":         "BF16",
    "int8":         "Uniform INT8",
    "int4":         "Uniform INT4",
    "gptq4":        "GPTQ (4-bit)",
    "awq4":         "AWQ (4-bit)",
    "spqr3":        "SpQR (3-bit)",
    "slim2":        "SliM-LLM (2-bit)",
    "smoothquant8": "SmoothQuant (8-bit)",
    "atom4":        "Atom (4-bit)",
    "mixed":        "LLM-QRP",
}

SKIP_IF_MISSING = {"gptq4", "awq4", "spqr3", "slim2", "smoothquant8", "atom4"}


def discover_results(results_dir):
    """Find all multi_dataset_benchmark.json under results_dir."""
    found = {}
    for model_dir in sorted(os.listdir(results_dir)):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        json_path = os.path.join(model_path, "quantize", "multi_dataset_benchmark.json")
        if os.path.isfile(json_path):
            found[model_dir] = json_path
    return found


def load_result(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_datasets(data):
    return list(data["dataset_results"].keys())


def write_per_model_table(data, tex_path):
    """Write a per-model baseline comparison table (one model only)."""
    model_name = data["model_name"]
    short_name = model_name.split("/")[-1]
    datasets = get_datasets(data)
    ds_displays = [data["dataset_results"][d]["display"] for d in datasets]

    present_methods = ["bf16", "int8", "int4", "mixed"]
    for eb in data.get("external_baselines", []):
        key = eb["key"]
        if key in METHOD_ORDER and key not in present_methods:
            present_methods.insert(-1, key)

    baseline_mb = data["baseline_size_mb"]

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{Quantization baselines for {short_name}}}",
        f"  \\label{{tab:bench_{short_name.replace('-', '_')}}}",
        r"  \begin{tabular}{lrrr" + "r" * len(datasets) + "}",
        r"    \toprule",
        f"    \\textbf{{Method}} & \\textbf{{Size (MB)}} & \\textbf{{VRAM (MB)}} "
        f"& \\textbf{{Compression}} & "
        + " & ".join(f"\\textbf{{{d}}}" for d in ds_displays)
        + r" \\",
        r"    \midrule",
    ]

    for key in present_methods:
        display = METHOD_DISPLAY.get(key, key)
        accs = data["dataset_results"].get(datasets[0], {}).get("accuracy", {})
        if key not in accs and key not in SKIP_IF_MISSING:
            continue
        if key not in accs:
            continue

        size_mb = data["dataset_results"][datasets[0]].get("size_mb", {}).get(key, 0)
        vram_mb = data["dataset_results"][datasets[0]].get("vram_mb", {}).get(key, 0)
        cmpr = baseline_mb / size_mb if size_mb > 0 else 0

        if key == "mixed":
            lines.append(r"    \midrule")

        vram_str = f"{vram_mb:.1f}" if vram_mb > 0 else "---"
        ds_vals = []
        for d in datasets:
            acc = data["dataset_results"][d]["accuracy"].get(key, 0)
            ds_vals.append(f"{acc:.4f}")
        ds_str = " & ".join(ds_vals)

        if key == "mixed":
            display = f"\\textbf{{{display}}}"
            ds_str_parts = ds_str.split(" & ")
            ds_str = " & ".join(f"\\textbf{{{v}}}" for v in ds_str_parts)

        lines.append(
            f"    {display} & {size_mb:.1f} & {vram_str} & {cmpr:.2f}x & {ds_str} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {tex_path}")


def write_combined_table(all_data, tex_path, csv_path):
    """Write a combined multi-model × multi-dataset table for the paper."""
    datasets_union = []
    for data in all_data.values():
        for d in get_datasets(data):
            if d not in datasets_union:
                datasets_union.append(d)
    ds_displays = []
    for d in datasets_union:
        for data in all_data.values():
            if d in data["dataset_results"]:
                ds_displays.append(data["dataset_results"][d]["display"])
                break
        else:
            ds_displays.append(d.upper())

    present_methods = ["bf16", "int8", "int4", "gptq4", "awq4", "spqr3", "slim2", "smoothquant8", "atom4", "mixed"]
    available_methods = []
    for m in present_methods:
        for data in all_data.values():
            accs = {}
            for d in get_datasets(data):
                accs.update(data["dataset_results"][d].get("accuracy", {}))
            if m in accs:
                available_methods.append(m)
                break

    n_ds = len(datasets_union)
    col_spec = "l l" + "r" * (2 + n_ds)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \caption{Quantization baselines across models and datasets. Best accuracy per model group is \textbf{bolded}.}",
        r"  \label{tab:combined_benchmark}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    \\textbf{{Model}} & \\textbf{{Method}} & \\textbf{{Size (MB)}} "
        f"& \\textbf{{Compression}} & "
        + " & ".join(f"\\textbf{{{d}}}" for d in ds_displays)
        + r" \\",
        r"    \midrule",
    ]

    first_model = True
    for model_key, data in all_data.items():
        short_name = data["model_name"].split("/")[-1]
        baseline_mb = data["baseline_size_mb"]
        datasets = get_datasets(data)

        best_accs = {}
        for d in datasets:
            accs = data["dataset_results"][d]["accuracy"]
            valid_accs = {k: v for k, v in accs.items() if k in available_methods}
            if valid_accs:
                best_accs[d] = max(valid_accs.values())

        if not first_model:
            lines.append(r"    \midrule")
        first_model = False

        model_rows = []
        for key in available_methods:
            display = METHOD_DISPLAY.get(key, key)
            accs_0 = data["dataset_results"].get(datasets[0], {}).get("accuracy", {})
            if key not in accs_0:
                continue

            size_mb = data["dataset_results"].get(datasets[0], {}).get("size_mb", {}).get(key, 0)
            cmpr = baseline_mb / size_mb if size_mb > 0 else 0

            ds_vals = []
            for d in datasets:
                acc = data["dataset_results"][d]["accuracy"].get(key, 0)
                if abs(acc - best_accs.get(d, -999)) < 1e-6 and key != "bf16":
                    ds_vals.append(f"\\textbf{{{acc:.4f}}}")
                else:
                    ds_vals.append(f"{acc:.4f}")
            ds_str = " & ".join(ds_vals)
            model_rows.append((key, display, size_mb, cmpr, ds_str))

        for i, (key, display, size_mb, cmpr, ds_str) in enumerate(model_rows):
            model_col = short_name if i == 0 else ""
            if key == "mixed":
                display = f"\\textbf{{{display}}}"
            lines.append(
                f"    {model_col} & {display} & {size_mb:.1f} & {cmpr:.2f}x & {ds_str} \\\\"
            )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {tex_path}")

    fieldnames = ["Model", "Method", "Size (MB)", "Compression"] + ds_displays
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_key, data in all_data.items():
            short_name = data["model_name"].split("/")[-1]
            baseline_mb = data["baseline_size_mb"]
            datasets = get_datasets(data)
            for key in available_methods:
                display = METHOD_DISPLAY.get(key, key)
                accs_0 = data["dataset_results"].get(datasets[0], {}).get("accuracy", {})
                if key not in accs_0:
                    continue
                size_mb = data["dataset_results"].get(datasets[0], {}).get("size_mb", {}).get(key, 0)
                cmpr = baseline_mb / size_mb if size_mb > 0 else 0
                row = {"Model": short_name, "Method": display,
                       "Size (MB)": f"{size_mb:.1f}", "Compression": f"{cmpr:.2f}x"}
                for d in datasets:
                    disp = data["dataset_results"][d]["display"]
                    row[disp] = f"{data['dataset_results'][d]['accuracy'].get(key, 0):.4f}"
                writer.writerow(row)
    print(f"  Wrote {csv_path}")


def write_dataset_wise_table(all_data, tex_path):
    """Reviewer request: Model | Dataset | BF16 | INT4 | GPTQ | AWQ | Mixed | LLM-QRP | VRAM | Compression"""
    methods = ["bf16", "int8", "int4", "gptq4", "awq4", "spqr3", "slim2", "smoothquant8", "atom4", "mixed"]
    active_methods = []
    for m in methods:
        for data in all_data.values():
            for d in get_datasets(data):
                if m in data["dataset_results"].get(d, {}).get("accuracy", {}):
                    active_methods.append(m)
                    break
            if m in active_methods:
                break

    method_headers = " & ".join(
        f"\\textbf{{{METHOD_DISPLAY.get(m, m)}}}" for m in active_methods
    )

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \caption{Dataset-wise accuracy comparison. Best per-model result \textbf{bolded}.}",
        r"  \label{tab:dataset_wise}",
        f"  \\begin{{tabular}}{{ll{'r' * len(active_methods)}rr}}",
        r"    \toprule",
        f"    \\textbf{{Model}} & \\textbf{{Dataset}} & {method_headers} "
        f"& \\textbf{{VRAM (MB)}} & \\textbf{{Compression}} \\\\",
        r"    \midrule",
    ]

    first_model = True
    for model_key, data in all_data.items():
        short_name = data["model_name"].split("/")[-1]
        baseline_mb = data["baseline_size_mb"]
        datasets = get_datasets(data)

        if not first_model:
            lines.append(r"    \midrule")
        first_model = False

        first_row = True
        for ds_key in datasets:
            ds_display = data["dataset_results"][ds_key]["display"]
            accs = data["dataset_results"][ds_key]["accuracy"]
            vram = data["dataset_results"][ds_key].get("vram_mb", {})
            size = data["dataset_results"][ds_key].get("size_mb", {})

            valid_accs = {m: accs.get(m, 0) for m in active_methods if m in accs}
            best_acc = max(valid_accs.values()) if valid_accs else 0

            model_col = short_name if first_row else ""
            first_row = False

            method_vals = []
            for m in active_methods:
                acc = accs.get(m, None)
                if acc is None:
                    method_vals.append("---")
                    continue
                display = METHOD_DISPLAY.get(m, m)
                is_best = abs(acc - best_acc) < 1e-6 and m != "bf16"
                is_qrp = m == "mixed"
                if is_qrp:
                    method_vals.append(f"\\textbf{{{acc:.4f}}}")
                elif is_best:
                    method_vals.append(f"\\textbf{{{acc:.4f}}}")
                else:
                    method_vals.append(f"{acc:.4f}")

            mixed_vram = vram.get("mixed", 0)
            mixed_size = size.get("mixed", 0)
            mixed_cmpr = baseline_mb / mixed_size if mixed_size > 0 else 0
            vram_str = f"{mixed_vram:.1f}" if mixed_vram > 0 else "---"

            vals_str = " & ".join(method_vals)
            lines.append(
                f"    {model_col} & {ds_display} & {vals_str} "
                f"& {vram_str} & {mixed_cmpr:.2f}x \\\\"
            )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-model benchmark results into combined paper tables."
    )
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Root results directory containing per-model subdirs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for combined tables (default: results/)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, "combined")
    os.makedirs(output_dir, exist_ok=True)

    found = discover_results(results_dir)
    if not found:
        print(f"No multi_dataset_benchmark.json found under {results_dir}")
        sys.exit(1)

    print(f"Found {len(found)} model result(s):")
    for name, path in found.items():
        print(f"  {name}: {path}")

    all_data = {}
    for name, path in found.items():
        try:
            data = load_result(path)
            all_data[name] = data
        except Exception as e:
            print(f"  WARNING: Failed to load {path}: {e}")

    if not all_data:
        print("No valid results loaded.")
        sys.exit(1)

    print(f"\nGenerating per-model tables...")
    for name, data in all_data.items():
        short_name = data["model_name"].split("/")[-1]
        tex_path = os.path.join(output_dir, f"benchmark_{short_name}.tex")
        write_per_model_table(data, tex_path)

    print(f"\nGenerating combined multi-model table...")
    combined_tex = os.path.join(output_dir, "benchmark_combined.tex")
    combined_csv = os.path.join(output_dir, "benchmark_combined.csv")
    write_combined_table(all_data, combined_tex, combined_csv)

    print(f"\nGenerating dataset-wise table...")
    dataset_tex = os.path.join(output_dir, "benchmark_dataset_wise.tex")
    write_dataset_wise_table(all_data, dataset_tex)

    print(f"\nDone! Output in {output_dir}")


if __name__ == "__main__":
    main()
