
MODEL_NAME=${1:-"HuggingFaceTB/SmolLM2-135M"}
OUTPUT_FOLDER=${2:-"D:/Downloads/Results"}
SAMPLES=${3:-1}

python -m qrp.benchmark.run_analysis --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES && \
python -m qrp.ablate.run_ablation --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER --samples $SAMPLES && \
python figures/create_figures.py --model-name $MODEL_NAME --output-folder $OUTPUT_FOLDER
