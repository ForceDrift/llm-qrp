# LLM-QRP: Quantization for Reasoning Preservation in Small LLMs

This software project accompanies the research paper *Quantization for Reasoning Preservation in Small LLMs* ([bibtex](#cite)).

<!-- <img width="1423" height="569" alt="image" src="https://github.com/user-attachments/assets/b114d9e4-47f0-4208-aae6-f2a0773346b8" /> -->

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
4. **Benchmark**: Evaluate the optimal mixed-precision model side-by-side against **BF16**, **Uniform INT8**, **Uniform INT4**, and (optionally) **GPTQ**, **AWQ**, **SpQR**, **SliM-LLM**, **SmoothQuant**, and **Atom** baselines across multiple datasets.

Quick summary of the main files in the repository:

* **Analysis & Ablation:**
	+ `qrp/benchmark/run_analysis.py`: Extracts per-layer thinking scores on a chosen dataset (supports `--signals` to select signal subsets and `--variant` for output subdirectories).
	+ `qrp/analysis/`: Sub-component SLED metric (`subcomponent_sled.py`, CoT-masked, attention/MLP residual decomposition, plus forward-only Information-Bottleneck Convergence Velocity via `entropy_velocity`), whole-layer SLED (`sled.py`), legacy injection-based entropy profiling (`entropy_by_layer.py`, deprecated for sub-component mode), score aggregation (`aggregate_scores.py`), ablation controller (`ablation_controller.py`).
	+ `qrp/ablate/run_ablation.py`: Ablates layers to measure their contribution.
	+ `qrp/ablate/run_signal_ablation.py`: Ablates over scoring-signal subsets (SLED, KL, Entropy) to compare reasoning accuracy vs compression trade-offs.
* **Quantization:**
	+ `qrp/quantize/allocation_core.py`: **Parameter-free** PCA fusion of profiling signals (first-principal-component loading weights on z-scored features) and an **exact 0-1 Multiple-Choice Knapsack** allocator over candidate bits `{2, 3, 4, 8, 16}` with non-linear precision fidelity (no λ-weighting, no min-max normalization, no manual layer percentiles).
+ `qrp/quantize/integer_quant.py`: `UniformIntLinear` (2/3/4-bit round-to-nearest) and `OutlierProtectedLinear` (Salient Outlier Channel Protection: top-0.1% channels kept as an unquantized BF16 sparse `W_fp16`, remainder low-bit).
	+ `qrp/quantize/allocate_mixed_precision.py`: CLI that profiles sub-component scores into a Pareto frontier of memory-budget-optimal BF16/INT8/FP4 allocations and selects the most efficient one above the accuracy floor.
	+ `qrp/quantize/quantizer.py`: `TargetedQuantizer`, per-layer *and* per-`(layer,{attn,mlp})` INT8/FP4 quantization via `bitsandbytes` (`quantize_layers` / `quantize_components`).
	+ `qrp/quantize/run_quantize_sweep.py`: Sweeps threshold-based quantization configs.
	+ `qrp/quantize/find_optimal_mixed_precision.py`: Legacy percentile-grid layer search (kept for backward compatibility).
	+ `qrp/quantize/export_quantized_model.py`: Exports the quantized checkpoint.
	+ `qrp/quantize/run_compression_report.py`: Compression report and chart (accuracy vs size).
	+ `qrp/quantize/run_multi_dataset_benchmark.py`: Multi-dataset benchmark producing Table 1 (BF16 / Uniform INT8 / Uniform INT4 / optional GPTQ, AWQ, SpQR, SliM-LLM, SmoothQuant, Atom / LLM-QRP), written to `benchmark_results.csv`, `benchmark_results.tex` and `multi_dataset_benchmark.json`.
	+ `qrp/external/gptq_baseline.py`: Adapter that runs the vendored [GPTQ](https://github.com/ist-daslab/gptq) implementation (`external/gptq`) uniformly over all decoder blocks, using GSM8K train sequences for calibration.
	+ `qrp/external/awq_baseline.py`: Adapter that runs the vendored [LLM-AWQ](https://github.com/mit-han-lab/llm-awq) implementation (`external/awq`) — activation-aware scaling, clipping, and grouped fake quantization — uniformly over all decoder blocks, device-agnostic.
	+ `qrp/external/spqr_baseline.py`: Adapter that runs the vendored [SpQR](https://github.com/Vahe1994/SpQR) implementation (`external/spqr`) — Hessian-based quantization with double-quantized group statistics and fp16 sparse outliers — uniformly over all decoder blocks, device-agnostic.
	+ `qrp/external/slim_baseline.py`: Adapter that runs the vendored [SliM-LLM](https://github.com/Aaronhuang-778/SliM-LLM) implementation (`external/slim-llm`) — salience-clustered mixed precision around a target bit-width with GPTQ reconstruction — uniformly over all decoder blocks, device-agnostic.
	+ `qrp/external/smoothquant_baseline.py`: Adapter that runs the vendored [SmoothQuant](https://github.com/mit-han-lab/smoothquant) implementation (`external/smoothquant`) — activation-scale-aware weight smoothing followed by per-channel absmax quantization — uniformly over all decoder blocks, device-agnostic.
	+ `qrp/external/atom_baseline.py`: Adapter that runs the vendored [Atom](https://github.com/efeslab/Atom) implementation (`external/atom`) — fine-grained group quantization with configurable symmetry and clip ratio — uniformly over all decoder blocks, device-agnostic.
* **Visualization:**
	+ `qrp/visualization/generate_report.py`: Aggregates JSON results into markdown reports and plots.
* **Shell Scripts (`scripts` directory):**
	+ `run_pipeline.sh`: Analysis + ablation loop over multiple datasets.
	+ `run_benchmarks.sh`: Unified decoding-method benchmarks (SLED vs baseline decoding).
	+ `run_test.ps1`: Full multi-model sweep (analysis, ablation, figures, quantization, mixed precision, reports).

> Layers with the lowest reasoning density are quantized first; the search in `find_optimal_mixed_precision.py` maximizes compression subject to retaining at least `--min-accuracy-floor` of the BF16 baseline accuracy.

### Sub-component bit-budget pipeline (evolved framework)

The framework generalizes from whole-layer percentile sweeps to a sub-component,
constrained-optimization pipeline. Each layer is split into an *attention* and an
*MLP* component; SLED scores are computed on CoT-masked token positions, the
**Information-Bottleneck Convergence Velocity** -- the average absolute
vocabulary-entropy transition across each sub-block over CoT tokens,
``DeltaH(l, c) = 1/|T_CoT| sum_t |H(P_in) - H(P_out)|`` -- is measured in the
same forward pass, and the two signals are fused by projecting their z-scored
feature vectors onto the first principal component (PCA loadings learned from
model dynamics), producing the criticality
``R_{l,c} = w_1 . tilde{S}_{SLED}(l,c) + w_2 . tilde{DeltaH}(l,c)``.

**Salient Outlier Channel Protection** extracts the top-0.1% highest-activation
weight channels per sub-matrix into an unquantized BF16 sparse matrix ``W_fp16``
(``outlier_channels.json`` emitted during profiling); only the remaining 99.9%
of each matrix is quantized to low precision.

Precision assignment is then solved as a **0-1 Multiple-Choice Knapsack** over
candidate bits ``{2, 3, 4, 8, 16}`` for a target average budget
``B_target`` (bits/param, e.g. 3.5):

    maximize    sum_{l,c,b} x_{l,c,b} . R_{l,c} . f_{l,c}(b)
    s.t.        1/P_total . sum_{l,c,b} x_{l,c,b} . P_{l,c} . b <= B_target,
                sum_b x_{l,c,b} = 1,   x_{l,c,b} in {0, 1}

with ``f(b)`` a non-linear precision fidelity anchored on the measured
convergence velocity. The step-wise percentile grid search is removed.

```bash
# 1. Profile per-(layer,{attn,mlp}) SLED + entropy scores over CoT reasoning tokens
python -m qrp.benchmark.run_analysis \
    --model-name HuggingFaceTB/SmolLM2-135M \
    --dataset gsm8k \
    --samples 50 \
    --output-folder ./results \
    --granularity subcomponent \
    --cot

# 2. Allocate precision by solving the bit-budget 0-1 knapsack; evaluate the frontier
python -m qrp.quantize.allocate_mixed_precision \
    --model-name HuggingFaceTB/SmolLM2-135M \
    --output-folder ./results \
    --samples 50 \
    --bits-per-param 3.5 \
    --min-accuracy-floor 0.5 \
    --bits-step 0.5
```

Outputs: per-component scores in `<output-folder>/<model>/<dataset>/subcomponent_scores.json`,
the protected channel lists in `.../outlier_channels.json`, and the selected
allocation (with full frontier) in
`<output-folder>/<model>/<dataset>/quantize/optimal_mixed_precision.json`.
Run `allocate_mixed_precision --help` for the budget/floor knobs; no percentiles
and no λ-weights are involved.

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

# 2. Benchmark against uniform baselines (Table 1); add --with-gptq / --with-awq / --with-spqr / --with-slim / --with-smoothquant / --with-atom for external baselines
python -m qrp.quantize.run_multi_dataset_benchmark \
    --model-name HuggingFaceTB/SmolLM2-135M \
    --output-folder ./results \
    --samples 100 \
    --datasets gsm8k,tfqa,mmlu \
    --with-gptq --gptq-bits 4 \
    --with-awq --awq-bits 4 \
    --with-spqr --spqr-bits 3 \
    --with-slim --slim-bits 2 \
    --with-smoothquant --smoothquant-bits 8 \
    --with-atom --atom-bits 4
```

Step 2 evaluates every dataset under up to ten conditions — the untouched **BF16** model, a **Uniform INT8** model (all layers 8-bit), a **Uniform INT4** model (all layers 4-bit), optional uniform **GPTQ**, **AWQ**, **SpQR**, **SliM-LLM**, **SmoothQuant**, and **Atom** models (all from the vendored reference implementations, calibrated on GSM8K train sequences), and the **LLM-QRP** optimal mix — reporting accuracy, estimated size (MB), compression ratio, and accuracy-per-MB efficiency for each. Results are stored in `./results/<model_name>/quantize/`.

### Signal-subset ablation (Reviewer response)

To compare reasoning accuracy vs compression/VRAM trade-off across different scoring-signal subsets (SLED only, KL only, Entropy only, SLED+KL, SLED+Entropy, KL+Entropy, and the full combination), run:

```bash
python -m qrp.ablate.run_signal_ablation \
    --model-name HuggingFaceTB/SmolLM2-135M \
    --dataset gsm8k \
    --samples 50 \
    --output-folder ./results
```

Optionally restrict to specific variants:

```bash
python -m qrp.ablate.run_signal_ablation \
    --model-name HuggingFaceTB/SmolLM2-135M \
    --dataset gsm8k \
    --samples 50 \
    --output-folder ./results \
    --variants sled_only kl_only full
```

The script runs the full pipeline (analysis → aggregation → optimal-mixed-precision search) for each variant, then prints a comparison table and saves `signal_ablation_summary.json` to the output folder. Each variant's intermediate files are stored under `<output_folder>/<model_name>/<variant>/`.

You can also run the analysis step for a single signal combination directly:

```bash
python -m qrp.benchmark.run_analysis \
    --model-name HuggingFaceTB/SmolLM2-135M \
    --dataset gsm8k \
    --samples 50 \
    --output-folder ./results \
    --variant sled_kl \
    --signals sled kl
```

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

If you find this repository helpful, please consider citing our work (not published yet):

```bibtex
@inproceedings{llmqrp2026,
  title={Quantization for Reasoning Preservation in Small LLMs},
  author={Iruku, Roshan and Dhillon, Gurshaan},
  year={2026}
}
```
