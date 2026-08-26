# Run full pipeline for Granite and Qwen only
param (
    [string]$OUTPUT_FOLDER = "./results",
    [int]$SAMPLES = 30,
    [string]$DATASETS = "gsm8k,tfqa,mmlu"
)

$MODELS = @(
    "ibm-granite/granite-4.0-350m-base",
    "Qwen/Qwen2.5-0.5B"
)

$PYTHON_EXE = ".\.venv\Scripts\python.exe"
if (-Not (Test-Path $PYTHON_EXE)) {
    $PYTHON_EXE = "python"
}

$ErrorActionPreference = "Continue"

foreach ($MODEL_NAME in $MODELS) {
    $SAFE_NAME = $MODEL_NAME.Replace("/", "_")
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "  Model: $MODEL_NAME"
    Write-Host "=========================================="

    # Step 1: Analysis
    Write-Host "`n>>> Step 1: run_analysis"
    & $PYTHON_EXE -m qrp.benchmark.run_analysis --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: run_analysis for $MODEL_NAME" -ForegroundColor Red
        continue
    }

    # Step 2: Ablation
    Write-Host "`n>>> Step 2: run_ablation"
    & $PYTHON_EXE -m qrp.ablate.run_ablation --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: run_ablation for $MODEL_NAME" -ForegroundColor Red
        continue
    }

    # Step 3: Figures
    Write-Host "`n>>> Step 3: create_figures"
    & $PYTHON_EXE figures/create_figures.py --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: create_figures for $MODEL_NAME (non-critical, continuing)" -ForegroundColor Yellow
    }

    # Step 4: Quantize sweep
    Write-Host "`n>>> Step 4: run_quantize_sweep"
    & $PYTHON_EXE -m qrp.quantize.run_quantize_sweep --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: run_quantize_sweep for $MODEL_NAME" -ForegroundColor Red
        continue
    }

    # Step 5: Find optimal mixed precision
    Write-Host "`n>>> Step 5: find_optimal_mixed_precision"
    & $PYTHON_EXE -m qrp.quantize.find_optimal_mixed_precision --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: find_optimal_mixed_precision for $MODEL_NAME" -ForegroundColor Red
        continue
    }

    # Step 6: Export model
    Write-Host "`n>>> Step 6: export_quantized_model"
    & $PYTHON_EXE -m qrp.quantize.export_quantized_model --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --threshold-4bit 1.0 --threshold-8bit 1.5
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: export_quantized_model for $MODEL_NAME (non-critical, continuing)" -ForegroundColor Yellow
    }

    # Step 7: Compression report
    Write-Host "`n>>> Step 7: run_compression_report"
    & $PYTHON_EXE -m qrp.quantize.run_compression_report --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: run_compression_report for $MODEL_NAME (non-critical, continuing)" -ForegroundColor Yellow
    }

    # Step 8: Full benchmark with all baselines
    Write-Host "`n>>> Step 8: run_multi_dataset_benchmark (FULL with all baselines)"
    & $PYTHON_EXE -m qrp.quantize.run_multi_dataset_benchmark --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES --datasets $DATASETS --with-gptq --gptq-bits 4 --with-awq --awq-bits 4 --with-spqr --spqr-bits 3 --with-slim --slim-bits 2 --with-smoothquant --smoothquant-bits 8 --with-atom --atom-bits 4
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: run_multi_dataset_benchmark for $MODEL_NAME" -ForegroundColor Red
        continue
    }

    Write-Host "`n  DONE: $MODEL_NAME" -ForegroundColor Green
}

Write-Host "`n=========================================="
Write-Host "  Regenerating combined tables..."
Write-Host "=========================================="
& $PYTHON_EXE -m qrp.quantize.aggregate_benchmarks --results-dir $OUTPUT_FOLDER

Write-Host "`n=========================================="
Write-Host "  All done!"
Write-Host "=========================================="
