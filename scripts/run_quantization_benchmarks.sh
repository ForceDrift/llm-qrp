#!/bin/bash

# Configuration
MODEL="HuggingFaceTB/SmolLM2-360M"
SCORES="results/layer_avg_scores.json"
LIMIT=10
OUT_DIR="results/sweep"
PYTHON_EXEC="./ai_env/Scripts/python.exe"

# Export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "===================================================="
echo "Starting Multi-Dataset Quantization Sweep"
echo "Model: $MODEL"
echo "Limit: $LIMIT"
echo "===================================================="

# --- GSM8K ---
echo "Running GSM8K Sweep..."
$PYTHON_EXEC -m qrp.benchmark.quantization_sweep \
    --scores-file "$SCORES" --model-name "$MODEL" --dataset "gsm8k" \
    --limit "$LIMIT" --thresholds 0.2 0.3 0.4 --bit-widths 4 \
    --output-folder "$OUT_DIR"

# --- StrategyQA (STRQA) ---
echo "Running StrategyQA Sweep..."
$PYTHON_EXEC -m qrp.benchmark.quantization_sweep \
    --scores-file "$SCORES" --model-name "$MODEL" --dataset "strqa" \
    --limit "$LIMIT" --thresholds 0.2 0.3 0.4 --bit-widths 4 \
    --output-folder "$OUT_DIR"

# --- TruthfulQA (TFQA) ---
echo "Running TruthfulQA Sweep..."
$PYTHON_EXEC -m qrp.benchmark.quantization_sweep \
    --scores-file "$SCORES" --model-name "$MODEL" --dataset "tfqa" \
    --limit "$LIMIT" --thresholds 0.2 0.3 0.4 --bit-widths 4 \
    --output-folder "$OUT_DIR"

echo "===================================================="
echo "Sweeps Complete! Generating Quantization Report..."
echo "===================================================="

$PYTHON_EXEC -m qrp.visualization.generate_report --results-dir "$OUT_DIR" --output-dir "results/reports"
