@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM  Full 100-sample multi-dataset benchmark suite (all 4 models)
REM  Runs BF16 / Uniform INT8 / Uniform INT4 / LLM-QRP (mixed) plus the 6
REM  external baselines (GPTQ4, AWQ4, SpQR3, SliM2, SmoothQuant8, Atom4) on
REM  GSM8K + TruthfulQA + MMLU at 100 samples each, for every model.
REM  Then regenerates the combined / per-model / dataset-wise tex tables.
REM ============================================================================

cd /d "%~dp0"

set "PY=ai_env\Scripts\python.exe"
set "OUT=results"
set "COMMON=--output-folder %OUT% --samples 100 --datasets gsm8k,tfqa,mmlu"
set "EXTRA=--with-gptq --with-awq --with-spqr --with-slim --with-smoothquant --with-atom"

if not exist "%PY%" (
    echo [ERROR] venv python not found: %PY%
    exit /b 1
)

echo.
echo ============================================================================
echo  Stage 0: sub-component profiling (100 samples each, needed by allocator)
echo ============================================================================
for %%M in (
    "HuggingFaceTB/SmolLM2-135M"
    "ibm-granite/granite-4.0-350m-base"
    "Qwen/Qwen2.5-0.5B"
    "LiquidAI/LFM2-350M"
) do (
    echo.
    echo  [profile] %%M
    "%PY%" -m qrp.benchmark.run_analysis --model-name "%%~M" --output-folder %OUT% --granularity subcomponent --cot --samples 100
    if errorlevel 1 goto :fail
)

echo.
echo ============================================================================
echo  Stage 1: sub-component MCKP allocation (50 evals, 8.0 bpw target, 3-bit caps)
echo ============================================================================
for %%M in (
    "HuggingFaceTB/SmolLM2-135M"
    "ibm-granite/granite-4.0-350m-base"
    "Qwen/Qwen2.5-0.5B"
    "LiquidAI/LFM2-350M"
) do (
    echo.
    echo  [allocate] %%M
    "%PY%" -m qrp.quantize.allocate_mixed_precision --model-name "%%~M" --output-folder %OUT% --samples 50 --bits-per-param 8.0 --low3-max-frac 0.0
    if errorlevel 1 goto :fail
)

echo.
echo ============================================================================
echo  Stage 2: full multi-dataset benchmark (100 samples x 3 datasets) per model
echo ============================================================================
for %%M in (
    "HuggingFaceTB/SmolLM2-135M"
    "ibm-granite/granite-4.0-350m-base"
    "Qwen/Qwen2.5-0.5B"
    "LiquidAI/LFM2-350M"
) do (
    echo.
    echo  [benchmark] %%M
    "%PY%" -m qrp.quantize.run_multi_dataset_benchmark --model-name "%%~M" %COMMON% %EXTRA%
    if errorlevel 1 goto :fail
)

echo.
echo ============================================================================
echo  Stage 3: regenerate combined / per-model / dataset-wise tex tables
echo ============================================================================
"%PY%" -m qrp.quantize.aggregate_benchmarks --results-dir %OUT%
if errorlevel 1 goto :fail

echo.
echo ============================================================================
echo  DONE - full 100-sample benchmark suite complete.
echo ============================================================================
exit /b 0

:fail
echo.
echo [ERROR] Step failed. Stopping suite.
exit /b 1
