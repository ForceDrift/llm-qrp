import os
import argparse
import json
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Create box plots from ablation metrics")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Model name evaluated")
    parser.add_argument("--output-folder", type=str, required=True, help="Base folder where results are saved")
    
    args = parser.parse_args()
    
    model_name_safe = args.model_name.replace("/", "_")
    base_dir = os.path.join(args.output_folder, model_name_safe)
    
    metrics_file = os.path.join(base_dir, "ablation", "ablation_metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Error: Could not find {metrics_file}")
        print("Please ensure you have run the ablation script first.")
        return

    with open(metrics_file, "r") as f:
        data = json.load(f)

    bottom_20_accs = [item[1] for item in data.get("bottom_20_ablation", [])]
    top_20_accs = [item[1] for item in data.get("top_20_ablation", [])]

    if not bottom_20_accs or not top_20_accs:
        print("Error: Required data keys missing or empty in ablation_metrics.json")
        return

    baseline = data.get("baseline_accuracy", None)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir)
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    
    plot_data = [bottom_20_accs, top_20_accs]
    labels = ['Bottom 20%\n(Lowest Thinking Layers)', 'Top 20%\n(Highest Thinking Layers)']
    
    box = plt.boxplot(plot_data, labels=labels, patch_artist=True)
    
    colors = ['#ADD8E6', '#FFB6C1']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for whisker, cap in zip(box['whiskers'], box['caps']):
        whisker.set(color='#333333', linewidth=1.5)
        cap.set(color='#333333', linewidth=1.5)

    for median in box['medians']:
        median.set(color='red', linewidth=2)

    if baseline is not None:
        plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline Prob ({baseline:.2%})')
        plt.legend(loc='upper right')

    plt.title(f'Performance Distribution Across Ablation Stages\nModel: {args.model_name}')
    plt.ylabel('Target Prob (exp(-loss))')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    out_path = os.path.join(figures_dir, "ablation_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Successfully generated box plot!")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
