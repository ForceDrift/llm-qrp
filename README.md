# LLM-QRP: Quantization for Reasoning Preservation in Small LLMs

This software project accompanies the research paper *Quantization for Reasoning Preservation in Small LLMs* ([bibtex](#cite)).

<img width="1423" height="569" alt="image" src="https://github.com/user-attachments/assets/b114d9e4-47f0-4208-aae6-f2a0773346b8" />

---

## Setup

1. Clone the Repository:

    ```bash
    git clone https://github.com/<owner>/llm-qrp
    cd llm-qrp
    ```

2. Install dependencies. Python 3.10+ is recommended, with PyTorch 2.5.1 + CUDA 12.1 for GPU machines.

    ```bash
    pip install -r requirements.txt
    ```

3. Models and datasets are downloaded automatically from Huggingface the first time you run any script. Set `HF_TOKEN` if you use gated models:

    ```bash
    export HF_TOKEN="your_token"
    ```

---

## Documentation

This repository contains the code for a research pipeline that profiles where reasoning happens inside small language models and quantizes around it. The workflow is:

1. **Profile Layers**: Extract layer-wise thinking scores (SLED-style disagreement and entropy metrics) on reasoning datasets.
2. **Ablate**: Verify which layers actually matter by measuring performance drops when they are skipped or degraded.
3. **Learn the Optimal Mix**: Sweep candidate mixed-precision configurations (BF16 / INT8 / FP4 per layer) and pick the one maximizing compression under an accuracy floor.
4. **Benchmark**: Evaluate the optimal mixed-precision model side-by-side against **BF16**, **Uniform INT8**, **Uniform INT4**, and (optionally) **GPTQ**, **AWQ**, and **SpQR** baselines across multiple datasets.

Quick summary of the main files in the repository:

* **Analysis & Ablation:**
	+ `qrp/benchmark/run_analysis.py`: Extracts per-layer thinking scores on a chosen dataset.
	+ `qrp/analysis/`: SLED metric (`sled.py`), entropy profiling (`entropy_by_layer.py`), score aggregation (`aggregate_scores.py`), ablation controller (`ablation_controller.py`).
	+ `qrp/ablate/run_ablation.py`: Ablates layers to measure their contribution.
* **Quantization:**
	+ `qrp/quantize/quantizer.py`: `TargetedQuantizer`, per-layer INT8/FP4 quantization via `bitsandbytes`.
	+ `qrp/quantize/run_quantize_sweep.py`: Sweeps threshold-based quantization configs.
	+ `qrp/quantize/find_optimal_mixed_precision.py`: Searches all candidate mixed-precision configs and selects the most efficient one.
	+ `qrp/quantize/export_quantized_model.py`: Exports the quantized checkpoint.
	+ `qrp/quantize/run_compression_report.py`: Compression report and chart (accuracy vs size).
	+ `qrp/quantize/run_multi_dataset_benchmark.py`: Multi-dataset benchmark producing Table 1 (BF16 / Uniform INT8 / Uniform INT4 / optional GPTQ, AWQ, SpQR / LLM-QRP), written to `benchmark_results.csv`, `benchmark_results.tex` and `multi_dataset_benchmark.json`.
	+ `qrp/external/gptq_baseline.py`: Adapter that runs the vendored [GPTQ](https://github.com/ist-daslab/gptq) implementation (`external/gptq`) uniformly over all decoder blocks, using GSM8K train sequences for calibration.
	+ `qrp/external/awq_baseline.py`: Adapter that runs the vendored [LLM-AWQ](https://github.com/mit-han-lab/llm-awq) implementation (`external/awq`) — activation-aware scaling, clipping, and grouped fake quantization — uniformly over all decoder blocks, device-agnostic.
	+ `qrp/external/spqr_baseline.py`: Adapter that runs the vendored [SpQR](https://github.com/Vahe1994/SpQR) implementation (`external/spqr`) — Hessian-based quantization with double-quantized group statistics and fp16 sparse outliers — uniformly over all decoder blocks, device-agnostic.
* **Visualization:**
	+ `qrp/visualization/generate_report.py`: Aggregates JSON results into markdown reports and plots.
* **Shell Scripts (`scripts` directory):**
	+ `run_pipeline.sh`: Analysis + ablation loop over multiple datasets.
	+ `run_benchmarks.sh`: Unified decoding-method benchmarks (SLED vs baseline decoding).
	+ `run_test.ps1`: Full multi-model sweep (analysis, ablation, figures, quantization, mixed precision, reports).

> Layers with the lowest reasoning density are quantized first; the search in `find_optimal_mixed_precision.py` maximizes compression subject to retaining at least `--min-accuracy-floor` of the BF16 baseline accuracy.

### Running the end-to-end pipeline

```bash
# Usage: ./scripts/run_pipeline.sh <model_name> <output_folder> <samples>
./scripts/run_pipeline.sh HuggingFaceTB/SmolLM2-135M ./results 100
```

This command will:
1. Run layer-wise analysis (thinking scores) on GSM8K, TruthfulQA, StrategyQA and FACTOR for the given model.
2. Run the ablation step to measure how much each layer contributes.
3. Store scores and ablation results in `<output_folder>/<model_name>/`.

Remember to switch device appropriately in your environment (`cuda` on GPU, `mps` on MacOS) when invoking the underlying modules directly.

### Finding the optimal mixed precision and benchmarking it

```bash
# 1. Search the optimal BF16/INT8/FP4 mix from aggregated layer scores
python -m qrp.quantize.find_optimal_mixed_precision \
    --model-name HuggingFaceTB/SmolLM2-135M \
    --output-folder ./results \
    --samples 100 \
    --min-accuracy-floor 0.5

# 2. Benchmark against uniform baselines (Table 1); add --with-gptq / --with-awq / --with-spqr for external baselines
python -m qrp.quantize.run_multi_dataset_benchmark \
    --model-name HuggingFaceTB/SmolLM2-135M \
    --output-folder ./results \
    --samples 100 \
    --datasets gsm8k,tfqa,mmlu \
    --with-gptq --gptq-bits 4 \
    --with-awq --awq-bits 4 \
    --with-spqr --spqr-bits 3
```

Step 2 evaluates every dataset under up to seven conditions — the untouched **BF16** model, a **Uniform INT8** model (all layers 8-bit), a **Uniform INT4** model (all layers 4-bit), optional uniform **GPTQ**, **AWQ**, and **SpQR** models (all from the vendored reference implementations, calibrated on GSM8K train sequences), and the **LLM-QRP** optimal mix — reporting accuracy, estimated size (MB), compression ratio, and accuracy-per-MB efficiency for each. Results are stored in `./results/<model_name>/quantize/`.

### Running the unified benchmarking suite

To test specific decoding methods directly against a dataset:

```bash
./scripts/run_benchmarks.sh
```
*Note: Make sure to modify the `MODEL`, `LIMIT` and `PYTHON_EXEC` parameters inside the script to your requirements before running.*

### Multi-model comprehensive sweep

For a complete evaluation of several models including ablation plots, quantization sweeps, optimal mixed-precision inference, compression reports and Table 1:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_test.ps1 -OUTPUT_FOLDER "./results" -SAMPLES 100 -DATASETS "gsm8k,tfqa"
```

The models swept are defined in `$MODELS` inside the script (by default SmolLM2-135M, Granite-4.0-350m, Qwen2.5-0.5B, LFM2-350M).

---

## Customizing Runs

All entry points are standard [argparse](https://docs.python.org/3/library/argparse.html) CLI programs, so every parameter can be overridden on the command line without touching the code.

1. **Change the model**: pass a different Huggingface hub id, e.g.

    ```bash
    python -m qrp.benchmark.run_analysis --model-name LiquidAI/LFM2-350M ...
    ```

2. **Choose datasets**: the multi-dataset benchmark accepts a comma-separated list among `gsm8k`, `tfqa`, `mmlu`:

    ```bash
    python -m qrp.quantize.run_multi_dataset_benchmark ... --datasets gsm8k,tfqa,mmlu
    ```

3. **Control the accuracy floor**: `--min-accuracy-floor` (fraction of baseline accuracy a candidate must retain, between 0.0 and 1.0).

4. **Tune SLED parameters**: if you are adjusting the SLED parameters in your evaluations, consider the following ranges for optimal reasoning preservation:
    * **Evolution Rate**: set within a range of **0.5 to 3**.
    * **Evolution Scale**: set values of **5, 10, or 20**.

---

## Acknowledgement

This codebase utilizes analytical metrics inspired by the official repos of [SLED](https://github.com/JayZhang42/SLED) and [LLM-AWQ](https://github.com/mit-han-lab/llm-awq).

## Cite

If you find this repository helpful, please consider citing our work (citation placeholder):

```bibtex
@inproceedings{
  llmqrp2026,
  title={Quantization for Reasoning Preservation in Small LLMs},
  author={Anonymous},
  year={2026}
}
```
