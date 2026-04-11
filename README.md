# LLM-QRP: Quantization for Reasoning Preservation in Small LLMs

---

The unofficial/official implementation for tracking and preserving reasoning capabilities during the quantization of small Large Language Models (LLMs).

## 📌 News
[2026.04.11] - Released the LLM-QRP multi-dataset pipeline and layer-wise entropy profiling code!

## 🧨 Why Choose LLM-QRP?
- <span style="color:#4285F4">Reasoning Preservation:</span> Unlike standard static quantization, LLM-QRP explicitly profiles layer disagreement and entropy to target precision drops where they hurt least.
- <span style="color:#4285F4">Mixed-Precision Tuning:</span> Identifies the optimal mix of 8-bit, 4-bit, and BF16 precision for each specific internal layer based on SLED and Entropy analysis.
- <span style="color:#4285F4">Task Versatility:</span> Built-in pipeline evaluation for complex reasoning datasets: GSM8K, TruthfulQA, StrategyQA, and FACTOR.
- <span style="color:#4285F4">Small LLM Focus:</span> Specifically tailored for high-efficiency small footprints (e.g., LFM2 models, SmolLM2 series).

## 🔮 Overview of LLM-QRP

We introduce a novel mixed-precision quantization methodology. Standard post-training quantization treats all layers uniformly, often crippling a model's multi-step reasoning. LLM-QRP leverages structural entropy and Skip-Layer Evaluation Decoding (SLED) style metrics to measure "thinking intensity" at each layer. Layers with high reasoning density are preserved in high precision (BF16 or 8-bit), while lower-density syntactic layers are pushed to 4-bit, maintaining near-baseline target probabilities with massive memory savings.

<img width="1423" height="569" alt="image" src="https://github.com/user-attachments/assets/b114d9e4-47f0-4208-aae6-f2a0773346b8" />


## 🛠 Installation
- **Python**: Recommended to use Python 3.10 or higher.
- **PyTorch**: Recommended PyTorch 2.5.1 with CUDA 12.1.
- **Dependencies**: 
  ```bash
  pip install -r requirements.txt
  ```

## 📈 Evaluation & Usage

Below we provide the necessary scripts to run our complete pipeline across your chosen models.

### End-to-End Pipeline
To run the automated analysis, ablation, and quantization benchmark loop across multiple datasets (GSM8K, TruthfulQA, StrategyQA, FACTOR):

```bash
# Usage: ./scripts/run_pipeline.sh <model_name> <output_folder> <samples>
./scripts/run_pipeline.sh HuggingFaceTB/SmolLM2-135M ./results 100
```

### Unified Benchmarking
To test specific decoding methods directly against a dataset:

```bash
./scripts/run_benchmarks.sh
```
*Note: Make sure to modify the `MODEL` and `LIMIT` parameters inside the script to your requirements before running.*

### Multi-Model Comprehensive Sweep
For a complete evaluation of several models including ablation plots and optimal mixed-precision inference:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_test.ps1 -OUTPUT_FOLDER "./results" -SAMPLES 100
```

## 💡 Important Recommendations
If you are adjusting the SLED parameters in your evaluations, consider the following parameters for optimal reasoning preservation:
- **Evolution Rate**: Set within a range of **0.5 to 3**. 
- **Evolution Scale**: Set values of **5, 10, or 20**. 

## Acknowledgement
This codebase utilizes analytical metrics inspired by the official repos of [SLED](https://github.com/JayZhang42/SLED) and [LLM-AWQ](https://github.com/mit-han-lab/llm-awq). 

## Citation
If you find this repository helpful, please consider citing our work (citation placeholder):
```bibtex
@inproceedings{
  llmqrp2026,
  title={Quantization for Reasoning Preservation in Small LLMs},
  author={Anonymous},
  year={2026}
}
```
