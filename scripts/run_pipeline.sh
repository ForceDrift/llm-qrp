#!/bin/bash

# Usage: ./scripts/run_pipeline.sh <model_name> <output_folder> <samples>
# Example: ./scripts/run_pipeline.sh HuggingFaceTB/SmolLM2-135M D:/Downloads/Results 100

MODEL_NAME=${1:-"HuggingFaceTB/SmolLM2-135M"}
OUTPUT_FOLDER=${2:-"./results"}
SAMPLES=${3:-1}

# Array of datasets to iterate over
DATASETS=("gsm8k" "tfqa" "strqa" "factor")

echo "Starting Analysis & Ablation Pipeline"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_FOLDER"
echo "Samples: $SAMPLES"
echo "---------------------------------------------------------"

for DATASET in "${DATASETS[@]}"
do
    echo "=================================================="
    echo "Processing dataset: $DATASET"
    echo "=================================================="
    
    echo "Step 1: Running Analysis (Generating thinking scores...)"
    python -m qrp.benchmark.run_analysis \
        --model-name "$MODEL_NAME" \
        --output-folder "$OUTPUT_FOLDER" \
        --dataset "$DATASET" \
        --samples "$SAMPLES"

    if [ $? -ne 0 ]; then
        echo "Error: Analysis failed for $DATASET. Exiting."
        exit 1
    fi

    echo "Step 2: Running Ablation (Testing performance drop...)"
    python -m qrp.ablate.run_ablation \
        --model-name "$MODEL_NAME" \
        --output-folder "$OUTPUT_FOLDER" \
        --samples "$SAMPLES"

    if [ $? -ne 0 ]; then
        echo "Error: Ablation failed for $DATASET. Exiting."
        exit 1
    fi
        
    echo "Successfully finished dataset: $DATASET"
    echo ""
done

echo "Pipeline completed for all datasets!"
