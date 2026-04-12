"""
Standard evaluation benchmarks for quantized LLMs.

Provides:
  - WikiText-2 perplexity (the universal quantization benchmark)
  - GSM8K exact-match accuracy (reasoning benchmark)

These are the metrics AWQ/GPTQ papers report, making results directly comparable.
"""

import math
import re

import torch
from datasets import load_dataset
from tqdm import tqdm


def evaluate_wikitext2_ppl(model, tokenizer, max_length=None, stride=None,
                            max_tokens=None):
    """
    Compute perplexity on WikiText-2 test set.

    This is the standard metric reported in GPTQ, AWQ, SqueezeLLM, etc.
    Uses a sliding window approach for sequences longer than model context.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        max_length: context window size (default: model's max position embeddings)
        stride: sliding window stride (default: max_length // 2)
        max_tokens: limit total tokens evaluated (for quick testing)

    Returns:
        float: perplexity value
    """
    model.eval()

    # Load WikiText-2 test set
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]

    seq_len = input_ids.size(1)

    if max_length is None:
        max_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_length = min(max_length, 2048)  # Cap to avoid OOM
    if stride is None:
        stride = max_length // 2

    nlls = []
    n_tokens = 0

    for begin_loc in tqdm(range(0, seq_len - 1, stride), desc="WikiText-2 PPL",
                          leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - begin_loc - 1

        chunk_ids = input_ids[:, begin_loc:end_loc].to(model.device)

        target_ids = chunk_ids.clone()
        # Mask out the overlapping context (only count unique tokens)
        if begin_loc > 0:
            overlap = max_length - stride
            target_ids[:, :overlap] = -100

        with torch.no_grad():
            outputs = model(chunk_ids, labels=target_ids)
            # loss is averaged over non-ignored tokens in the batch
            neg_log_likelihood = outputs.loss

        # Count actual tokens scored (non -100)
        scored = (target_ids != -100).sum().item()
        nlls.append(neg_log_likelihood.float() * scored)
        n_tokens += scored

        if end_loc >= seq_len:
            break

    if n_tokens == 0:
        return float('inf')

    avg_nll = torch.stack(nlls).sum() / n_tokens
    ppl = torch.exp(avg_nll).item()
    return ppl


def evaluate_gsm8k_exact_match(model, tokenizer, n_samples=100,
                                 max_new_tokens=256):
    """
    Compute exact-match accuracy on GSM8K (grade school math).

    This is the standard reasoning benchmark. We generate answers and extract
    the final numerical answer using the #### delimiter pattern.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        n_samples: number of test questions to evaluate
        max_new_tokens: max tokens to generate per question

    Returns:
        dict with keys: accuracy, correct, total, results
    """
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main", split="test")
    samples = list(dataset)[:n_samples]

    correct = 0
    total = 0
    results = []

    for item in tqdm(samples, desc="GSM8K Exact-Match", leave=False):
        question = item["question"]
        answer_text = item["answer"]

        # Extract ground truth number
        gt_match = re.search(r"####\s*([\d,.\-]+)", answer_text)
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()

        # Generate
        prompt = (
            f"Solve the following math problem step by step. "
            f"End your answer with '#### <number>'.\n\n"
            f"Question: {question}\nAnswer:"
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        # Extract predicted answer
        pred = _extract_number(generated)

        # Compare
        is_correct = False
        try:
            if pred and abs(float(pred) - float(gt)) < 1e-6:
                is_correct = True
        except (ValueError, TypeError):
            is_correct = pred == gt

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "gt": gt,
            "pred": pred,
            "correct": is_correct,
            "generated": generated[:200],
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def _extract_number(text):
    """Extract the final numerical answer from generated text."""
    # Try #### pattern first (trained format)
    matches = re.findall(r"####\s*([\d,.\-]+)", text)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Try "the answer is" pattern
    matches = re.findall(r"the answer is\s*([\d,.\-]+)", text, re.IGNORECASE)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Fallback: last number in the text
    matches = re.findall(r"[\-]?\d[\d,.]*", text)
    if matches:
        return matches[-1].replace(",", "").strip()

    return ""


def evaluate_target_prob(model, tokenizer, dataset_name="gsm8k", n_samples=10):
    """
    Compute target probability score (QRP's existing metric).
    Included for comparison with previous results.
    
    Returns exp(-avg_cross_entropy_loss) over answer tokens.
    """
    model.eval()

    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        pairs = [(x["question"], x["answer"]) for x in list(ds)[:n_samples]]
    elif dataset_name == "tfqa":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        pairs = [(x["question"], x["best_answer"]) for x in list(ds)[:n_samples]]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    total_loss = 0.0
    valid = 0

    for question, answer in tqdm(pairs, desc=f"Target Prob ({dataset_name})",
                                   leave=False):
        prompt = f"Question: {question}\nAnswer: Let's think step by step\n"
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
