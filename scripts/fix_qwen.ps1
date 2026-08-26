# Targeted script: only runs the failed steps for Qwen
param (
    [string]$OUTPUT_FOLDER = "./results",
    [int]$SAMPLES = 30,
    [string]$DATASETS = "gsm8k,tfqa,mmlu"
)

$PYTHON_EXE = "python"

$MODEL_NAME = "Qwen/Qwen2.5-0.5B"
$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "  Targeted Qwen fix: Steps 1, 5, 8"
Write-Host "=========================================="

# Step 1: Analysis (split-phase to avoid OOM) — run for ALL 3 datasets
Write-Host "`n>>> Step 1a: run_analysis (gsm8k)"
& $PYTHON_EXE -m qrp.benchmark.run_analysis --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES --dataset gsm8k
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: run_analysis gsm8k" -ForegroundColor Red
    exit 1
}

Write-Host "`n>>> Step 1b: run_analysis (truthfulqa)"
& $PYTHON_EXE -m qrp.benchmark.run_analysis --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES --dataset truthfulqa
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: run_analysis truthfulqa" -ForegroundColor Red
    exit 1
}

Write-Host "`n>>> Step 1c: run_analysis (mmlu)"
& $PYTHON_EXE -m qrp.benchmark.run_analysis --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES --dataset mmlu
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: run_analysis mmlu" -ForegroundColor Red
    exit 1
}

# Step 5: Find optimal mixed precision (needs aggregated_scores.json from ALL datasets)
Write-Host "`n>>> Step 5: find_optimal_mixed_precision"
& $PYTHON_EXE -m qrp.quantize.find_optimal_mixed_precision --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: find_optimal_mixed_precision" -ForegroundColor Red
    exit 1
}

# Step 8: Full benchmark with all baselines
Write-Host "`n>>> Step 8: run_multi_dataset_benchmark (FULL)"
& $PYTHON_EXE -m qrp.quantize.run_multi_dataset_benchmark --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES --datasets $DATASETS --with-gptq --gptq-bits 4 --with-awq --awq-bits 4 --with-spqr --spqr-bits 3 --with-slim --slim-bits 2 --with-smoothquant --smoothquant-bits 8 --with-atom --atom-bits 4
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: run_multi_dataset_benchmark" -ForegroundColor Red
    exit 1
}

# Regenerate tables
Write-Host "`n>>> Regenerating combined tables"
& $PYTHON_EXE -m qrp.quantize.aggregate_benchmarks --results-dir $OUTPUT_FOLDER

Write-Host "`n=========================================="
Write-Host "  Done!"
Write-Host "=========================================="
