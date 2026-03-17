# Selective Quantization Performance (SLED/Entropy)

| Model                      | Dataset   | Condition                   |   Threshold |   Bit-Width |   Quantized Layers |   Accuracy |   Drop |
|:---------------------------|:----------|:----------------------------|------------:|------------:|-------------------:|-----------:|-------:|
| HuggingFaceTB/SmolLM2-360M | gsm8k     | baseline                    |       nan   |         nan |                  0 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | gsm8k     | baseline                    |       nan   |         nan |                  0 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | gsm8k     | threshold=0.20, bit_width=4 |         0.2 |           4 |                  0 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | gsm8k     | threshold=0.30, bit_width=4 |         0.3 |           4 |                 23 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | gsm8k     | threshold=0.30, bit_width=4 |         0.3 |           4 |                 23 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | gsm8k     | threshold=0.40, bit_width=4 |         0.4 |           4 |                 29 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | gsm8k     | threshold=0.50, bit_width=4 |         0.5 |           4 |                 30 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | strqa     | baseline                    |       nan   |         nan |                  0 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | strqa     | threshold=0.20, bit_width=4 |         0.2 |           4 |                  0 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | strqa     | threshold=0.30, bit_width=4 |         0.3 |           4 |                 23 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | strqa     | threshold=0.40, bit_width=4 |         0.4 |           4 |                 29 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | tfqa      | baseline                    |       nan   |         nan |                  0 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | tfqa      | threshold=0.20, bit_width=4 |         0.2 |           4 |                  0 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | tfqa      | threshold=0.30, bit_width=4 |         0.3 |           4 |                 23 |          0 |      0 |
| HuggingFaceTB/SmolLM2-360M | tfqa      | threshold=0.40, bit_width=4 |         0.4 |           4 |                 29 |          0 |      0 |
