import os
import torch
import torch.nn.functional as F
import argparse
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from qrp.analysis.sled import SLED_Decoded

class UnifiedBenchmarkRunner:
    def __init__(self, model_name, device="cuda", num_gpus=1):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Determine dtype
        self.dtype = torch.bfloat16 if "cuda" in device else torch.float32
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=self.dtype,
            device_map="auto" if num_gpus > 1 else None
        )
        if num_gpus == 1:
            self.model.to(self.device)
            
        self.sled_decoder = SLED_Decoded(model_name, self.device)
        # SLED handles the model loading
        self.model = self.sled_decoder.model

    def load_dataset(self, name, limit=None):
        if name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test")
            prompts = [{"question": x["question"], "answer": x["answer"]} for x in ds]
        elif name == "strqa":
            ds = load_dataset("voidful/StrategyQA", split="train") # STRQA test set meta-data usually private
            prompts = [{"question": x["question"], "answer": x["answer"]} for x in ds]
        elif name == "tfqa":
            ds = load_dataset("truthful_qa", "generation", split="validation")
            prompts = [{"question": x["question"], "answer": x["best_answer"]} for x in ds]
        else:
            raise ValueError(f"Unsupported dataset: {name}")
            
        if limit:
            prompts = prompts[:limit]
        return prompts

    def extract_answer(self, dataset_name, model_output, gt_answer=None):
        if dataset_name == "gsm8k":
            # GT extraction
            match = re.search(r"####\s*([\d,\.\-]+)", str(gt_answer)) if gt_answer else None
            gt = match.group(1).replace(",", "").strip() if match else str(gt_answer).strip()
            
            # Model extraction
            patterns = [r"the answer is\s*([\d,\.\-]+)", r"####\s*([\d,\.\-]+)", r"([\d,\.\-]+)\s*$"]
            pred = ""
            for p in patterns:
                m = re.findall(p, model_output, re.IGNORECASE)
                if m: pred = m[-1].replace(",", "").strip(); break
            if not pred:
                nums = re.findall(r"[\-]?\d[\d,\.]*", model_output)
                if nums: pred = nums[-1].replace(",", "").strip()
            
            try:
                correct = abs(float(pred) - float(gt)) < 1e-6
            except:
                correct = pred == gt
            return pred, gt, correct

        elif dataset_name == "strqa":
            pred = "yes" if "yes" in model_output.lower() else "no"
            gt = str(gt_answer).lower().strip()
            return pred, gt, pred == gt

        elif dataset_name == "tfqa":
            # For TruthfulQA, we often use semantic similarity or exact match for 'best_answer'
            # Simplified: check if gt_answer is in model_output
            pred = model_output.strip()
            gt = str(gt_answer).strip()
            return pred, gt, gt.lower() in pred.lower()

        return model_output, gt_answer, False

    def run(self, dataset_name, decoding_method, limit=10, output_path=None, **kwargs):
        samples = self.load_dataset(dataset_name, limit)
        results = []
        correct_count = 0

        for sample in tqdm(samples, desc=f"Evaluating {dataset_name} ({decoding_method})"):
            question = sample["question"]
            gt_answer = sample["answer"]
            
            prompt = f"Question: {question}\nAnswer:"
            if dataset_name == "gsm8k":
                prompt = f"Solve the following math problem step by step. End your answer with '#### <number>'.\n\n{prompt}"

            if decoding_method == "SLED":
                model_output = self.sled_decoder.generate(
                    prompt, 
                    max_new_tokens=256,
                    evolution_rate=kwargs.get("evolution_rate", 2),
                    evolution_scale=kwargs.get("evolution_scale", 10)
                )
                if prompt in model_output:
                    model_output = model_output.split("Answer:")[-1]
            else:
                raise ValueError(f"Unknown decoding method: {decoding_method}")
            
            pred, gt, is_correct = self.extract_answer(dataset_name, model_output, gt_answer)
            if is_correct:
                correct_count += 1
                
            results.append({
                "question": question,
                "gt": gt,
                "pred": pred,
                "model_output": model_output,
                "correct": is_correct
            })

        accuracy = correct_count / len(samples) if samples else 0
        
        summary = {
            "model": self.model_name,
            "dataset": dataset_name,
            "decoding": decoding_method,
            "accuracy": accuracy,
            "count": len(samples),
            "results": results
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
        
        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "strqa", "tfqa"], required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--decoding_method", type=str, choices=["SLED"], required=True)
    parser.add_argument("--evolution_rate", type=float, default=2.0)
    parser.add_argument("--evolution_scale", type=int, default=10)
    
    args = parser.parse_args()
    
    runner = UnifiedBenchmarkRunner(args.model_name, num_gpus=args.num_gpus)
    summary = runner.run(
        args.dataset, 
        args.decoding_method, 
        limit=args.limit, 
        output_path=args.output_path,
        evolution_rate=args.evolution_rate,
        evolution_scale=args.evolution_scale
    )
    
    print(f"\nBenchmark Complete!")
    print(f"Accuracy: {summary['accuracy']:.4f}")
