from datasets import load_dataset
from qrp.analysis.run_analysis import SLEDEntropyAnalyzer
import os
import json
import torch

MAX_SAMPLES = 1
SEED = 42

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=SEED).select(range(MAX_SAMPLES))

    analyzer = SLEDEntropyAnalyzer()

    save_folder = "results"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "gsm8k_1_test.json")

    results = {}

    for i, sample in enumerate(dataset):
        prompt = sample["question"]
        print(f"\nRunning sample {i+1}/{MAX_SAMPLES}")

        result = analyzer.run(prompt)

        results[f"sample_{i}"] = {
            "prompt": prompt,
            "result": result
        }

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nFinished. Results saved to {save_path}")
