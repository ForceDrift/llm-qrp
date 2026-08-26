import json

models = {
    'SmolLM2-135M': 'results/HuggingFaceTB_SmolLM2-135M/quantize/multi_dataset_benchmark.json',
    'LFM2-350M': 'results/LiquidAI_LFM2-350M/quantize/multi_dataset_benchmark.json',
    'Qwen2.5-0.5B': 'results/Qwen_Qwen2.5-0.5B/quantize/multi_dataset_benchmark.json',
    'granite-4.0-350m': 'results/ibm-granite_granite-4.0-350m-base/quantize/multi_dataset_benchmark.json',
}
datasets = ['gsm8k', 'tfqa', 'mmlu']
ds_labels = {'gsm8k': 'GSM8K', 'tfqa': 'TruthfulQA', 'mmlu': 'MMLU'}

for name, path in models.items():
    data = json.load(open(path))
    results = data['results']
    bf16 = {d: results['BF16']['datasets'][d]['accuracy'] for d in datasets}
    qrp = {d: results['LLM-QRP (quantized)']['datasets'][d]['accuracy'] for d in datasets}
    print(f'\n=== {name} ===')
    for d in datasets:
        delta = (qrp[d] - bf16[d]) / bf16[d] * 100 if bf16[d] != 0 else 0
        best_name, best_val = '', 0
        for mname, mdata in results.items():
            if mname in ('BF16', 'LLM-QRP (quantized)'):
                continue
            if d in mdata['datasets']:
                v = mdata['datasets'][d]['accuracy']
                if v > best_val:
                    best_val = v
                    best_name = mname
        delta_vs_best = (qrp[d] - best_val) / best_val * 100 if best_val != 0 else 0
        print(f'  {ds_labels[d]}: QRP={qrp[d]:.4f}  BF16={bf16[d]:.4f} ({delta:+.1f}%)  best_baseline={best_name}={best_val:.4f} ({delta_vs_best:+.1f}%)')
