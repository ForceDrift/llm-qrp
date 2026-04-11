import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def generate_report(results_dir, output_dir):
    benchmark_data = []
    quant_data = []
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r") as f:
                        res = json.load(f)
                        
                        if "decoding" in res:
                            benchmark_data.append({
                                "Model": res.get("model", "Unknown"),
                                "Dataset": res.get("dataset", "Unknown"),
                                "Decoding": res.get("decoding", "Unknown"),
                                "Accuracy": res.get("accuracy", 0.0),
                                "Count": res.get("count", 0)
                            })
                        elif "results" in res and any("threshold" in r for r in res["results"]):
                            dataset = res.get("config", {}).get("dataset", "Unknown")
                            model = res.get("config", {}).get("model_name", "Unknown")
                            for r in res["results"]:
                                quant_data.append({
                                    "Model": model,
                                    "Dataset": dataset,
                                    "Condition": r.get("condition", "Unknown"),
                                    "Threshold": r.get("threshold"),
                                    "Bit-Width": r.get("bit_width"),
                                    "Quantized Layers": r.get("num_layers_quantized", 0),
                                    "Accuracy": r.get("accuracy", 0.0),
                                    "Drop": r.get("accuracy_drop", 0.0)
                                })
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    os.makedirs(output_dir, exist_ok=True)

    if benchmark_data:
        df = pd.DataFrame(benchmark_data)
        pivot_df = df.pivot_table(index=["Model", "Dataset"], columns="Decoding", values="Accuracy")
        md_table = pivot_df.to_markdown()
        
        with open(os.path.join(output_dir, "benchmark_summary.md"), "w") as f:
            f.write("# Decoding Method Comparison\n\n")
            f.write(md_table)
            f.write("\n")

        datasets = df["Dataset"].unique()
        for ds in datasets:
            ds_df = df[df["Dataset"] == ds]
            plt.figure(figsize=(10, 6))
            plt.bar(ds_df["Decoding"], ds_df["Accuracy"], color=['#4285F4', '#34A853', '#FBBC05'])
            plt.title(f"Performance Comparison on {ds.upper()}")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1.0)
            for i, acc in enumerate(ds_df["Accuracy"]):
                plt.text(i, acc + 0.02, f"{acc:.2%}", ha='center', fontweight='bold')
            plt.savefig(os.path.join(output_dir, f"{ds}_decoding_comparison.png"))
            plt.close()

    if quant_data:
        q_df = pd.DataFrame(quant_data)
        q_df = q_df.sort_values(by=["Dataset", "Bit-Width", "Threshold"], na_position='first')
        md_quant = q_df.to_markdown(index=False)
        with open(os.path.join(output_dir, "quantization_summary.md"), "w") as f:
            f.write("# Selective Quantization Performance (SLED/Entropy)\n\n")
            f.write(md_quant)
            f.write("\n")

        datasets = q_df["Dataset"].unique()
        for ds in datasets:
            ds_df = q_df[(q_df["Dataset"] == ds) & (q_df["Threshold"].notnull())]
            if ds_df.empty: 
                continue
            
            plt.figure(figsize=(10, 6))
            for bw in ds_df["Bit-Width"].unique():
                bw_df = ds_df[ds_df["Bit-Width"] == bw]
                plt.plot(bw_df["Threshold"], bw_df["Accuracy"], marker='o', label=f"{bw}-bit")
            
            baseline_rows = q_df[(q_df["Dataset"] == ds) & (q_df["Condition"] == "baseline")]
            if not baseline_rows.empty:
                baseline_acc = baseline_rows["Accuracy"].iloc[0]
                plt.axhline(y=baseline_acc, color='r', linestyle='--', label="Baseline (FP16)")
            
            plt.title(f"Quantization Threshold vs Accuracy ({ds.upper()})")
            plt.xlabel("Score Threshold (Higher = More Quantization)")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{ds}_quantization_sweep.png"))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/reports")
    args = parser.parse_args()
    generate_report(args.results_dir, args.output_dir)
