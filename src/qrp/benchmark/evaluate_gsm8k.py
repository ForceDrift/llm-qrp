"""
evaluate_gsm8k.py — Shared GSM8K answer extraction and accuracy measurement.

Reused by both thinking_ablation_benchmark.py and quantization_sweep.py.

Ground-truth format in the HuggingFace GSM8K dataset:
    answer field ends with "#### {numeric_answer}"

Model output: greedy-decoded text — we extract the last numeric value.
"""

import re
import torch
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def extract_gt_answer(gt_answer_str: str) -> str:
    """Extract the numeric ground truth after '####'.

    Example:
        "She had 5 apples. #### 5" → "5"
    """
    match = re.search(r"####\s*([\d,\.\-]+)", gt_answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def extract_model_answer(model_output: str) -> str:
    """Extract the last numeric value from model's greedy output.

    Looks for patterns like "the answer is 42", "= 42", or just the last
    standalone number in the output.
    """
    # Try "the answer is X" pattern first
    patterns = [
        r"(?:the answer is|answer:|=)\s*([\d,\.\-]+)",
        r"####\s*([\d,\.\-]+)",          # model may reproduce GSM8K format
        r"([\d,\.\-]+)\s*$",             # last number in output
    ]
    for pattern in patterns:
        matches = re.findall(pattern, model_output, re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "").strip()

    # Fallback: find all numbers and return the last one
    all_numbers = re.findall(r"[\-]?\d[\d,\.]*", model_output)
    if all_numbers:
        return all_numbers[-1].replace(",", "").strip()
    return ""


def answers_match(pred: str, gt: str) -> bool:
    """Numeric equality check (handles floats and ints)."""
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        return pred.strip() == gt.strip()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_gsm8k(
    model,
    tokenizer,
    samples: List[dict],
    device: str = "cpu",
    max_new_tokens: int = 256,
    verbose: bool = False,
) -> Tuple[float, List[dict]]:
    """Run greedy-decode evaluation on a list of GSM8K samples.

    Args:
        model: a HuggingFace CausalLM (already loaded, possibly with hooks).
        tokenizer: matching tokenizer.
        samples: list of dicts with keys "question" and "answer" (raw GSM8K rows).
        device: torch device string.
        max_new_tokens: generation budget.
        verbose: if True, print per-sample results.

    Returns:
        (accuracy, per_sample_results)
        accuracy: float in [0, 1]
        per_sample_results: list of dicts with keys:
            id, question, gt_answer, model_output, pred_answer, correct
    """
    model.eval()

    correct = 0
    results = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        gt_raw = sample["answer"]
        gt_answer = extract_gt_answer(gt_raw)

        prompt = (
            "Solve the following math problem step by step. "
            "End your answer with '#### <number>'.\n\n"
            f"Question: {question}\nAnswer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy
                pad_token_id=tokenizer.eos_token_id,
            )

        # decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        model_output = tokenizer.decode(new_ids, skip_special_tokens=True)

        pred_answer = extract_model_answer(model_output)
        is_correct = answers_match(pred_answer, gt_answer)

        if is_correct:
            correct += 1

        result = {
            "id": i,
            "question": question,
            "gt_answer": gt_answer,
            "model_output": model_output,
            "pred_answer": pred_answer,
            "correct": is_correct,
        }
        results.append(result)

        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"[{status}] Sample {i}: gt={gt_answer}, pred={pred_answer}")

    accuracy = correct / len(samples) if samples else 0.0
    return accuracy, results
