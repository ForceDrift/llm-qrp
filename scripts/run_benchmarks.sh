#!/bin/bash

# Configuration
MODEL="HuggingFaceTB/SmolLM2-360M"
LIMIT=10
OUT_DIR="results/benchmarks"
PYTHON_EXEC="./ai_env/Scripts/python.exe"

# Export PYTHONPATH to include src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "===================================================="
echo "Starting Unified Benchmarking Suite"
echo "Model: $MODEL"
echo "Limit: $LIMIT"
echo "===================================================="

# --- GSM8K ---
echo "Running GSM8K..."

$PYTHON_EXEC -m qrp.benchmark.run_unified \
    --model-name "$MODEL" --dataset "gsm8k" \
    --decoding_method "SLED" --limit "$LIMIT" \
    --evolution_rate 2 --evolution_scale 10 \
    --output-path "$OUT_DIR/gsm8k_sled.json"


# --- StrategyQA (STRQA) ---
echo "Running StrategyQA..."

$PYTHON_EXEC -m qrp.benchmark.run_unified \
    --model-name "$MODEL" --dataset "strqa" \
    --decoding_method "SLED" --limit "$LIMIT" \
    --evolution_rate 2 --evolution_scale 10 \
    --output-path "$OUT_DIR/strqa_sled.json"


# --- TruthfulQA (TFQA) ---
echo "Running TruthfulQA..."

$PYTHON_EXEC -m qrp.benchmark.run_unified \
    --model-name "$MODEL" --dataset "tfqa" \
    --decoding_method "SLED" --limit "$LIMIT" \
    --evolution_rate 2 --evolution_scale 10 \
    --output-path "$OUT_DIR/tfqa_sled.json"

echo "===================================================="
echo "Benchmarks Complete! Generating Report..."
echo "===================================================="

$PYTHON_EXEC -m qrp.visualization.generate_report --results-dir "$OUT_DIR" --output-dir "results/reports"
