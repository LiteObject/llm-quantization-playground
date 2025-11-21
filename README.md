# LLM Quantization Playground

A comprehensive benchmarking suite for comparing different LLM quantization methods: **FP16**, **INT8**, **GPTQ (4-bit)**, **AWQ (4-bit)**, and **GGML/GGUF**. This project helps you understand the speed vs. accuracy tradeoffs of model quantization techniques.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [What is Quantization?](#what-is-quantization)
- [Supported Quantization Methods](#supported-quantization-methods)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results & Benchmarks](#results--benchmarks)
- [Understanding the Tradeoffs](#understanding-the-tradeoffs)
- [Project Structure](#project-structure)
- [Docker Support](#docker-support)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## What is Quantization?

**Quantization** is a technique to reduce the memory footprint and computational requirements of neural networks by representing weights and activations with lower precision data types.

### Why Quantize?

- **Reduced Memory**: 4-bit models can be 4x smaller than FP16 models
- **Faster Inference**: Lower precision math is faster on modern GPUs
- **Accessibility**: Run larger models on consumer hardware (e.g., RTX 4090)
- **Cost Efficiency**: Lower cloud computing costs

### The Tradeoff

Quantization involves a **speed vs. accuracy tradeoff**:
- Higher precision (FP16) → Better accuracy, slower, more memory
- Lower precision (4-bit) → Lower accuracy, faster, less memory

This project helps you **measure and visualize** these tradeoffs!

---

## Supported Quantization Methods

| Method | Precision | Library | Description |
|--------|-----------|---------|-------------|
| **FP16** | 16-bit | PyTorch | Half-precision floating point (baseline) |
| **INT8** | 8-bit | bitsandbytes | Integer quantization with minimal accuracy loss |
| **GPTQ** | 4-bit | auto-gptq | Post-training quantization optimized for GPUs |
| **AWQ** | 4-bit | autoawq | Activation-aware Weight Quantization |
| **GGML** | 4-bit | llama-cpp-python | CPU-optimized quantization format |

### Quick Comparison

```
FP16:  [████████████████] 100% accuracy, slowest, 16GB memory
INT8:  [██████████████░░] ~98% accuracy, 2x faster, 8GB memory
GPTQ:  [████████████░░░░] ~95% accuracy, 3x faster, 4GB memory
AWQ:   [████████████░░░░] ~95% accuracy, 3x faster, 4GB memory
GGML:  [███████████░░░░░] ~93% accuracy, CPU-friendly, 4GB memory
```

*(Actual results vary by model and task)*

---

## Features

- **Unified Interface**: Load any quantization type with a single function
- **Speed Benchmarking**: Measure tokens/second and latency
- **Accuracy Evaluation**: QA accuracy and perplexity metrics
- **Rich Visualizations**: Beautiful charts comparing all methods
- **Docker Support**: Reproducible environment with GPU support
- **Detailed Logging**: Track memory usage and performance
- **Configurable**: Easy customization via config files
- **Production Ready**: Clean code with proper error handling

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 12.1+ (for GPU support)
- 16GB+ GPU memory recommended (RTX 4090, A100, etc.)

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/LiteObject/llm-quantization-playground.git
cd llm-quantization-playground

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with GPU support (optional)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
```

### Option 2: Docker Installation

```bash
# Build Docker image
docker build -t llm-quant-playground .

# Run container with GPU support
docker run --gpus all -it -v $(pwd)/results:/workspace/results llm-quant-playground
```

---

## Quick Start


### Run Complete Pipeline

```bash
# Run all benchmarks and evaluations
python scripts/run_all.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Visualize results
python scripts/visualize_results.py --results-dir results
```

### Benchmark Speed Only

```bash
python -m quantization.benchmark_speed \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --types fp16 int8 \
    --num-prompts 20
```

### Evaluate Accuracy Only

```bash
python -m quantization.evaluate_accuracy \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --types fp16 int8 gptq awq
```

---

## Usage

### Loading Models

```python
from quantization import load_model, QuantizationType

# Load FP16 model
model_fp16 = load_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_type=QuantizationType.FP16
)

# Load INT8 model
model_int8 = load_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_type=QuantizationType.INT8
)

# Generate text
response = model_fp16.generate("Explain quantum computing", max_new_tokens=100)
```

### Running Benchmarks

```python
from quantization import run_all_benchmarks

results = run_all_benchmarks(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_prompts=10,
    max_new_tokens=100
)
```

### Custom Evaluation

```python
from quantization import evaluate_model, load_model, QuantizationType

model = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", QuantizationType.GPTQ)

qa_pairs = [
    {"question": "What is Python?", "answer": "programming language"}
]

results = evaluate_model(model, evaluation_dataset=qa_pairs)
print(f"Accuracy: {results['qa_accuracy_percent']}%")
```

---

## Results & Benchmarks

### Example Output

```
BENCHMARK COMPARISON
================================================================================
Model Type   Tokens/Sec   Latency (s)    Memory (GB)
--------------------------------------------------------------------------------
INT8         45.23        2.2115         3.45
FP16         28.67        3.4876         6.52
GPTQ         52.18        1.9165         3.21
AWQ          51.34        1.9478         3.18
GGML         38.92        2.5692         3.15
================================================================================
```

### Visualization Examples

The `visualize_results.py` script generates:

1. **Speed Comparison**: Bar charts of tokens/second and latency
2. **Memory Usage**: GPU memory consumption by quantization type
3. **Accuracy Comparison**: QA accuracy and perplexity scores
4. **Speed vs Accuracy Tradeoff**: Scatter plot showing the pareto frontier

Example visualizations:

![Speed Comparison](results/speed_comparison.png)
![Accuracy Comparison](results/accuracy_comparison.png)
![Speed vs Accuracy Tradeoff](results/speed_accuracy_tradeoff.png)

---

## Understanding the Tradeoffs

### When to Use Each Method

**FP16 (Baseline)**
- ✅ Best accuracy
- ✅ Good GPU support
- ❌ Highest memory usage
- 💡 Use when: Accuracy is critical, have large GPU memory

**INT8 (bitsandbytes)**
- ✅ ~2x memory reduction
- ✅ Minimal accuracy loss (<2%)
- ✅ Good balance
- 💡 Use when: Want efficiency without sacrificing accuracy

**GPTQ (4-bit)**
- ✅ ~4x memory reduction
- ✅ Excellent GPU performance
- ⚠️ Moderate accuracy loss (3-5%)
- 💡 Use when: Need to run larger models on limited GPU

**AWQ (4-bit)**
- ✅ ~4x memory reduction
- ✅ Preserves activation outliers
- ✅ Better accuracy than GPTQ
- 💡 Use when: Want best 4-bit accuracy

**GGML (4-bit)**
- ✅ CPU-friendly
- ✅ Cross-platform compatibility
- ⚠️ Slower than GPU methods
- 💡 Use when: No GPU available or need portability

### Real-World Recommendations

| GPU Memory | Recommended Methods | Model Size Examples |
|------------|--------------------|--------------------|
| 8GB | INT8, GGML | 7B models |
| 12GB | INT8, GPTQ, AWQ | 13B models |
| 16GB | All methods | 7-13B models |
| 24GB+ | FP16, INT8 | 13-30B models |

---

## Project Structure

```
llm-quantization-playground/
│
├── quantization/              # Core quantization library
│   ├── __init__.py           # Package initialization
│   ├── quant_loader.py       # Model loading for all quantization types
│   ├── benchmark_speed.py    # Speed benchmarking utilities
│   ├── evaluate_accuracy.py  # Accuracy evaluation utilities
│   └── utils.py              # Shared helper functions
│
├── scripts/                   # Executable scripts
│   ├── run_all.py            # Run complete benchmark pipeline
│   └── visualize_results.py  # Generate visualization plots
│
├── configs/                   # Configuration files
│   ├── default_config.json   # Default settings
│   ├── prompts.txt           # Benchmark prompts
│   └── evaluation_dataset.json # QA pairs for evaluation
│
├── models/                    # (Optional) Downloaded models
│
├── results/                   # Output directory
│   ├── benchmark_results.json
│   ├── evaluation_results.json
│   └── *.png                 # Generated plots
│
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── .dockerignore            # Docker ignore file
└── README.md                # This file
```

---

## Docker Support

### Build and Run

```bash
# Build image
docker build -t llm-quant-playground .

# Run with GPU support
docker run --gpus all -it \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/models:/workspace/models \
    llm-quant-playground bash

# Inside container
python scripts/run_all.py
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  quant-playground:
    build: .
    runtime: nvidia
    volumes:
      - ./results:/workspace/results
      - ./models:/workspace/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

---

## Configuration

### Config File Example

`configs/default_config.json`:

```json
{
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "quantization_types": ["fp16", "int8", "gptq"],
    "benchmark_settings": {
        "num_prompts": 10,
        "max_new_tokens": 100,
        "num_warmup": 2
    },
    "evaluation_settings": {
        "use_qa_pairs": true,
        "calculate_perplexity": true
    }
}
```

### Command Line Arguments

```bash
python scripts/run_all.py --help

Options:
  --model TEXT              Model name or path
  --types [fp16|int8|gptq|awq|ggml]
                           Quantization types to test
  --num-prompts INTEGER    Number of prompts for benchmarking
  --max-tokens INTEGER     Maximum tokens to generate
  --output-dir TEXT        Output directory for results
  --skip-benchmark         Skip speed benchmarking
  --skip-evaluation        Skip accuracy evaluation
```

---

## Contributing

Contributions are welcome! Here are some ways to contribute:

- Report bugs and issues
- Suggest new features or quantization methods
- Improve documentation
- Add more evaluation metrics
- Enhance visualizations

### Development Setup

```bash
# Clone and install in editable mode
git clone https://github.com/LiteObject/llm-quantization-playground.git
cd llm-quantization-playground
pip install -e .

# Run tests (if available)
pytest tests/
```

---

## Future Improvements

- [ ] **More Quantization Methods**
  - GGUF support
  - SmoothQuant
  - LLM.int8() variants
  
- [ ] **Advanced Metrics**
  - Token-level perplexity
  - BLEU/ROUGE scores
  - Human evaluation integration
  
- [ ] **Model Support**
  - Llama 2/3 models
  - Mistral models
  - GPT-J/GPT-NeoX
  
- [ ] **Performance Optimizations**
  - Flash Attention integration
  - vLLM backend option
  - Multi-GPU support
  
- [ ] **Enhanced Visualizations**
  - Interactive Plotly dashboards
  - Real-time monitoring
  - Pareto frontier analysis
  
- [ ] **Additional Features**
  - Quantization from scratch (not just pre-quantized)
  - Mixed-precision configurations
  - Calibration dataset customization
  - A/B testing framework

---

## Resources & References

### Documentation
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)

### Related Projects
- [TheBloke's Quantized Models](https://huggingface.co/TheBloke)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
