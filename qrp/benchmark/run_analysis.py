import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from qrp.analysis.entropy_by_layer import EntropyByLayer
from qrp.analysis.sled import SLED_Decoded
from qrp.analysis.subcomponent_sled import COMPONENTS, SubComponentSLED


class SLEDEntropyAnalyzer:
    VALID_SIGNALS = {"sled", "kl", "entropy"}

    def __init__(self, model_name="HuggingFaceTB/SmolLM2-360M", dataset="gsm8k",
                 signals=None):
        self.model_name = model_name
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if signals is None:
            signals = ["sled", "kl", "entropy"]
        invalid = set(signals) - self.VALID_SIGNALS
        if invalid:
            raise ValueError(f"Invalid signals: {invalid}. Choose from {self.VALID_SIGNALS}")
        self.signals = list(signals)

        needs_sled = "sled" in self.signals or "kl" in self.signals
        needs_entropy = "entropy" in self.signals

        self.sled_model = None
        self.entropy_model = None
        if needs_sled:
            self.sled_model = SLED_Decoded(model_name, self.device)
        if needs_entropy:
            self.entropy_model = EntropyByLayer(model_name, device=self.device)

    def _minMaxScale(self, data_dict):
        values = list(data_dict.values())
        if not values:
            return data_dict

        minVal = np.min(values)
        maxVal = np.max(values)
        if maxVal == minVal:
            return {k: 0.0 for k in data_dict}

        return {k: (v - minVal) / (maxVal - minVal) for k, v in data_dict.items()}

    def _compute_sled(self, prompt):
        layers_to_test = list(range(self.sled_model.layers))
        disagreement_results = self.sled_model.layer_disagreement(
            prompt,
            evolution_scale=5,
            candidate_premature_layers=layers_to_test
        )

        layer_disagreement = defaultdict(list)
        for token_data in disagreement_results:
            for layer_idx, scores in token_data.items():
                layer_disagreement[layer_idx].append(scores.mean().item())

        layer_disagreement_mean = {
            f"layer_{k}": float(torch.tensor(v).mean())
            for k, v in layer_disagreement.items()
        }
        return self._minMaxScale(layer_disagreement_mean)

    def _compute_kl(self, prompt):
        kl_prev = self.sled_model.kl_between_current_prev(prompt)
        kl_prev_dict = {
            f"layer_{i}": float(kl.item())
            for i, kl in enumerate(kl_prev, 1)
        }
        return self._minMaxScale(kl_prev_dict)

    def _compute_entropy(self, prompt):
        entropy_results_raw = self.entropy_model.runStructuralAnalysis(prompt)
        entropy_dict = {f"layer_{res['layer']}": res['deltaH'] for res in entropy_results_raw}
        return self._minMaxScale(entropy_dict)

    def run(self, prompt):
        signal_values = {}
        if "sled" in self.signals:
            signal_values["sled"] = self._compute_sled(prompt)
        if "kl" in self.signals:
            signal_values["kl"] = self._compute_kl(prompt)
        if "entropy" in self.signals:
            signal_values["entropy"] = self._compute_entropy(prompt)

        all_keys = set()
        for sv in signal_values.values():
            all_keys.update(sv.keys())
        all_keys = sorted(all_keys)

        combined = {}
        for k in all_keys:
            contributing = []
            for sv in signal_values.values():
                if k in sv:
                    contributing.append(sv[k])
            if contributing:
                combined[k] = sum(contributing) / len(contributing)

        return combined


def load_prompts(dataset_name):
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [x["question"] for x in ds]
    elif dataset_name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        return [x["question"] for x in ds]
    elif dataset_name.lower() == "truthfulqa":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        return [x["question"] for x in ds]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_samples(dataset_name):
    """Load (question, answer) pairs for sub-component / CoT profiling."""
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [{"question": x["question"], "answer": x["answer"]} for x in ds]
    elif dataset_name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        samples = []
        for x in ds:
            choices = " ".join(x["choices"])
            answer = x["choices"][x["answer"]] if x["answer"] < len(x["choices"]) else ""
            samples.append({"question": f"{x['question']}\nChoices: {choices}", "answer": answer})
        return samples
    elif dataset_name.lower() == "truthfulqa":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        return [{"question": x["question"], "answer": x["best_answer"]} for x in ds]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


class SubComponentAnalyzer:
    """CoT-masked sub-component profiling: SLED + Information-Bottleneck Velocity.

    Produces per-(layer, component) scores from a single residual-stream
    forward pass (Section 3.1/3.2):

      * ``sled``   -- average cosine between evolution gradients and mature
                      logit divergence over CoT reasoning positions;
      * ``entropy``-- Information-Bottleneck Convergence Velocity:

                         DeltaH(l, c) = 1/|T_CoT| * sum_t |H(P_in) - H(P_out)|

                      the average absolute Shannon-entropy transition of the
                      projected vocabulary distribution across the sub-block
                      over CoT tokens.  Forward-only; no synthetic 4-bit
                      injection noise.

    When written to ``subcomponent_scores.json`` the entropy signal is stored
    under the ``delta_h`` key (schema consumed by ``allocate_mixed_precision``).
    """

    VALID_SIGNALS = {"sled", "entropy"}

    def __init__(self, model_name="HuggingFaceTB/SmolLM2-360M", dataset="gsm8k",
                 signals=None, device=None):
        self.model_name = model_name
        self.dataset = dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if signals is None:
            signals = ["sled", "entropy"]
        invalid = set(signals) - self.VALID_SIGNALS
        if invalid:
            raise ValueError(f"Invalid signals for sub-component mode: {invalid}")
        self.signals = list(signals)

        # Both signals come from the same residual capture, so a single model
        # load provides sled + entropy velocity (unlike whole-layer mode).
        self.sled = SubComponentSLED(model_name, self.device)
        self.tokenizer = self.sled.tokenizer

    def _prompt(self, sample):
        q = sample["question"]
        if self.dataset == "gsm8k":
            return f"Question: {q}\nAnswer: Let's think step by step\n"
        return f"Question: {q}\nAnswer: "

    def run(self, sample, cot=True, cot_start=None, outlier_share=0.001):
        prompt = self._prompt(sample)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        answer = sample.get("answer", "")
        full_text = prompt + (answer if answer else "")
        full_ids = self.tokenizer(full_text, return_tensors="pt").input_ids.to(self.device)
        if cot_start is None:
            cot_start = prompt_ids.shape[-1]
        cot_start = cot_start if cot else None

        # One forward pass back-to-back: SLED + entropy velocity + the
        # top-0.1% salient channels for micro-clipping.
        profiled = self.sled.profile(full_ids, cot_start, outlier_share=outlier_share)
        signal_values = {}
        if "sled" in self.signals:
            signal_values["sled"] = {k: float(v) for k, v in profiled["sled"].items()}
        if "entropy" in self.signals:
            signal_values["entropy"] = {k: float(v) for k, v in profiled["entropy"].items()}
        outliers = {k: list(v) for k, v in profiled["outlier_channels"].items()}
        return signal_values, outliers, profiled["channels_per_component"]


def aggregate_scores(results_data, output_file):
    layer_sums = defaultdict(float)
    layer_counts = defaultdict(int)

    for item in results_data:
        analysis = item.get("analysis", {})
        for layer_key, score in analysis.items():
            layer_sums[layer_key] += score
            layer_counts[layer_key] += 1
            
    avg_scores = {}
    for layer_key, total in layer_sums.items():
        avg_scores[layer_key] = total / layer_counts[layer_key]
        
    with open(output_file, "w") as f:
        json.dump(avg_scores, f, indent=2)


def aggregate_subcomponent_scores(results_data, model_name, signals, cot_masked, output_file):
    """Aggregate per-(layer, component) raw signals across samples.

    Writes the schema consumed by ``allocate_mixed_precision.py``: one entry
    per component with signal means, ready for the parameter-free fusion and
    bit-budget allocation.
    """
    sums = {k: defaultdict(float) for k in signals}
    counts = defaultdict(int)
    for item in results_data:
        comp = item.get("components", {})
        for cid in comp:
            counts[cid] += 1
            for name in signals:
                sums[name][cid] += comp[cid].get(name, 0.0)

    components = []
    for cid in sorted(counts):
        layer, c = cid.split(".", 1)
        comp = {"id": cid, "layer": int(layer), "component": c}
        for name in signals:
            out_key = "delta_h" if name == "entropy" else name
            comp[out_key] = sums[name][cid] / counts[cid]
        comp["samples"] = counts[cid]
        components.append(comp)

    out = {
        "model_name": model_name,
        "dataset": None,
        "signals": signals,
        "cot_masked": cot_masked,
        "components": components,
    }
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    return components


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLED Entropy Analysis")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M", help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to evaluate (gsm8k, mmlu)")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to analyze")
    parser.add_argument("--output-folder", type=str, required=True, help="Folder to save results")
    parser.add_argument("--signals", type=str, nargs="+", default=["sled", "kl", "entropy"],
                        choices=["sled", "kl", "entropy"],
                        help="Signals to compute and combine")
    parser.add_argument("--variant", type=str, default=None,
                        help="Optional variant name for output subdirectory (e.g. 'sled_only')")
    parser.add_argument("--granularity", type=str, choices=["layer", "subcomponent"], default="layer",
                        help="layer = whole-layer signals (SLED/KL/entropy); subcomponent = per (layer,{attn,mlp})")
    parser.add_argument("--cot", action="store_true",
                        help="Mask scoring to intermediate CoT reasoning tokens (subcomponent mode only)")
    parser.add_argument("--outlier-share", type=float, default=0.001,
                        help="Fraction of highest-activation channels protected to BF16 (subcomponent mode only)")

    args = parser.parse_args()

    model_name_safe = args.model_name.replace("/", "_")

    if args.variant:
        output_dir = os.path.join(args.output_folder, model_name_safe, args.variant)
    else:
        output_dir = os.path.join(args.output_folder, model_name_safe, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    if args.granularity == "subcomponent":
        invalid = set(args.signals) - SubComponentAnalyzer.VALID_SIGNALS
        if invalid:
            print(f"[subcomponent] dropping signals unsupported at this granularity: {sorted(invalid)}")
        signals = [s for s in args.signals if s in SubComponentAnalyzer.VALID_SIGNALS]
        if not signals:
            raise SystemExit("Subcomponent mode requires at least one of: sled, entropy")

        analyzer = SubComponentAnalyzer(model_name=args.model_name, dataset=args.dataset,
                                        signals=signals)
        samples = load_samples(args.dataset)[: args.samples]

        results = []
        outlier_votes = defaultdict(list)
        channel_counts = {}
        for idx, sample in enumerate(tqdm(samples, desc="Subcomponent profiling", unit="prompt")):
            signal_values, outliers, channels_per_component = analyzer.run(
                sample, cot=args.cot, outlier_share=args.outlier_share
            )
            for cid, chans in outliers.items():
                outlier_votes[cid].append(set(chans))
            channel_counts[f"channels_per_component"] = channels_per_component
            components = {}
            for name in signals:
                components.update({k: v for k, v in signal_values[name].items()})
            results.append({"id": idx, "prompt": sample.get("question"), "components": components})

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        sc_file = os.path.join(output_dir, "subcomponent_scores.json")
        aggregate_subcomponent_scores(results, args.model_name, signals, args.cot, sc_file)
        print(f"\nAggregated sub-component scores saved to: {sc_file}")
        print(f"(signals={signals}, cot_masked={args.cot})")

        if outlier_votes:
            # Majority-vote the protected channels across samples, per component.
            outlier_channels = {}
            for cid in sorted(outlier_votes):
                votes = outlier_votes[cid]
                count = len(votes)
                tally = {}
                for chans in votes:
                    for ch in chans:
                        tally[ch] = tally.get(ch, 0) + 1
                threshold = math.ceil(count * 0.5)
                outlier_channels[cid] = sorted(ch for ch, v in tally.items() if v >= threshold)
            oc_file = os.path.join(output_dir, "outlier_channels.json")
            with open(oc_file, "w") as f:
                json.dump({
                    "model_name": args.model_name,
                    "share": args.outlier_share,
                    "channels_per_component": channel_counts.get("channels_per_component"),
                    "outlier_channels": outlier_channels,
                }, f, indent=2)
            print(f"Salient outlier channels saved to: {oc_file}")

        # Layer-level aggregation (mean over components) for downstream compatibility.
        sc = json.load(open(sc_file))
        lsum, lcnt = defaultdict(float), defaultdict(int)
        for c in sc["components"]:
            layer = str(c["layer"])
            lcnt[layer] += 1
            lsum[layer] += sum(c.get(s, 0.0) for s in signals) / len(signals)
        with open(os.path.join(output_dir, "aggregated_scores.json"), "w") as f:
            json.dump({f"layer_{k}": lsum[k] / lcnt[k] for k in lcnt}, f, indent=2)
        print("Saved layer-averaged aggregated_scores.json for compatibility.")
        raise SystemExit(0)

    analyzer = SLEDEntropyAnalyzer(model_name=args.model_name, dataset=args.dataset,
                                   signals=args.signals)

    prompts = load_prompts(args.dataset)
    prompts = prompts[:args.samples]

    results = []

    for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts", unit="prompt")):
        analysis = analyzer.run(prompt)
        results.append({
            "id": idx,
            "prompt": prompt,
            "analysis": analysis
        })

    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    aggregated_file = os.path.join(output_dir, "aggregated_scores.json")
    aggregate_scores(results, aggregated_file)
    print(f"Aggregated scores saved to: {aggregated_file}")