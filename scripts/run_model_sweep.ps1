# Multi-model sweep: runs the full LLM-QRP pipeline for each model,
# then aggregates all results into a single combined LaTeX/CSV table.
#
# Models:
#   HuggingFaceTB/SmolLM2-135M
#   ibm-granite/granite-4.0-350m-base
#   Qwen/Qwen2.5-0.5B
#   LiquidAI/LFM2-350M

param (
    [string]$OUTPUT_FOLDER = "D:/Downloads/Results",
    [int]$SAMPLES = 50,
    [string]$DATASETS = "gsm8k,tfqa"
)

$ErrorActionPreference = "Stop"

$PYTHON_EXE = ".\.venv\Scripts\python.exe"
if (-Not (Test-Path $PYTHON_EXE)) {
    $PYTHON_EXE = "python"
}

$MODELS = @(
    "HuggingFaceTB/SmolLM2-135M",
    "ibm-granite/granite-4.0-350m-base",
    "Qwen/Qwen2.5-0.5B",
    "LiquidAI/LFM2-350M"
)

Write-Host "=================================================="
Write-Host "  LLM-QRP Multi-Model Sweep"
Write-Host "  Output:   $OUTPUT_FOLDER"
Write-Host "  Samples:  $SAMPLES"
Write-Host "  Datasets: $DATASETS"
Write-Host "  Models:"
foreach ($m in $MODELS) { Write-Host "    - $m" }
Write-Host "=================================================="

foreach ($MODEL_NAME in $MODELS) {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "  Running pipeline for: $MODEL_NAME"
    Write-Host "=========================================="

    & .\scripts\run_test.ps1 `
        -MODEL_NAME   $MODEL_NAME `
        -OUTPUT_FOLDER $OUTPUT_FOLDER `
        -SAMPLES       $SAMPLES `
        -DATASETS      $DATASETS

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Pipeline failed for $MODEL_NAME — continuing with next model..."
        continue
    }

    Write-Host "  Finished: $MODEL_NAME"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  Aggregating all models into final table"
Write-Host "=========================================="

$MODEL_LIST = $MODELS -join ","
& $PYTHON_EXE -m qrp.quantize.aggregate_benchmark_table `
    --output-folder $OUTPUT_FOLDER `
    --models        $MODEL_LIST `
    --datasets      $DATASETS

Write-Host ""
Write-Host "Done! Combined table saved to $OUTPUT_FOLDER/combined_benchmark.*"
