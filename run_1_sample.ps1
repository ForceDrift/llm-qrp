$ErrorActionPreference = "Stop"
$MODEL = "HuggingFaceTB/SmolLM2-135M"
Write-Host ">>> Running Analysis..."
.\.venv\Scripts\python.exe -m qrp.benchmark.run_analysis --model-name $MODEL --output-folder ./results --samples 1 --dataset gsm8k

Write-Host ">>> Running GPTQ Benchmark (with LLM-QRP configs)..."
.\.venv\Scripts\python.exe -m qrp.quantize.run_gptq_benchmark --model-name $MODEL --output-folder ./results --samples 1 --wikitext-tokens 256
