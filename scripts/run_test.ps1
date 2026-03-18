param (
    [string]$MODEL_NAME = "HuggingFaceTB/SmolLM2-135M",
    [string]$OUTPUT_FOLDER = "D:/Downloads/Results",
    [int]$SAMPLES = 1
)

$ErrorActionPreference = "Stop"

$PYTHON_EXE = ".\.venv\Scripts\python.exe"
if (-Not (Test-Path $PYTHON_EXE)) {
    $PYTHON_EXE = "python"
}

& $PYTHON_EXE -m qrp.benchmark.run_analysis --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
& $PYTHON_EXE -m qrp.ablate.run_ablation --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
& $PYTHON_EXE figures/create_figures.py --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER
& $PYTHON_EXE -m qrp.quantize.run_quantize_sweep --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
& $PYTHON_EXE -m qrp.quantize.find_optimal_mixed_precision --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES
# & $PYTHON_EXE -m qrp.quantize.export_quantized_model --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --threshold-4bit 1.0 --threshold-8bit 1.5
