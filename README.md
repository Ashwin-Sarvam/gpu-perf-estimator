# LLM Inference Performance & Cost Calculator

An interactive web tool to estimate LLM inference performance metrics and costs on NVIDIA Hopper & Blackwell GPUs. This calculator uses roofline models to provide accurate estimates for prefill latency, decode throughput, and economic analysis.

## Features

- **Multi-GPU Support**: Calculate performance across multiple nodes with tensor parallelism
- **Roofline Model**: Memory-bound and compute-bound performance analysis
- **MoE Support**: Accurate calculations for Mixture of Experts models
- **Cost Analysis**: Input/output token cost estimation
- **Multiple Precisions**: Support for FP16, FP8, and FP4 inference
- **Hardware Support**: H100, H200, and B200 GPUs

## Usage

### Basic Usage

1. **Select Model**: Enter a Hugging Face model ID or manually input model parameters
2. **Configure Hardware**: Choose your GPU type and number of nodes
3. **Set Parameters**: Configure sequence lengths, precision, and tensor parallelism
4. **Calculate**: Click "Calculate Performance" to get results

### Model Configuration

#### Using Hugging Face Models
- Enter the model ID (e.g., `deepseek-ai/DeepSeek-V3.1`)
- The tool will automatically fetch model parameters from the config.json

#### Manual Configuration
- Check "Manually enter model parameters?"
- Enter the following parameters:
  - Total Parameters
  - Hidden Size
  - Number of Layers
  - Number of Attention Heads
  - Number of Key-Value Heads (for GQA)
  - Number of Experts (for MoE models)
  - Active Experts per Token (for MoE models)

### Key Parameters

- **Max Input Sequence Length**: Length of input prompts (default: 2048)
- **Max Output Sequence Length**: Maximum tokens to generate (default: 1024)
- **Target Hardware**: GPU type (H100 SXM, H200 SXM, B200 SXM)
- **Inference Precision**: FP16, FP8, or FP4
- **Number of Nodes**: Total number of GPU nodes
- **Tensor Parallelism**: Number of GPUs per model replica (1-8)

### Understanding Results

#### System & Batching
- **Max Batch Size (per Replica)**: Maximum concurrent requests per model replica
- **Model Weights Memory**: Total memory required for model parameters
- **Max KV Cache Memory**: Memory allocated for attention key-value caches
- **Tensor Parallelism**: Total number of model replicas across all nodes

#### Performance
- **Prefill Latency (TTFT)**: Time to first token for input processing
- **Decode Throughput (per Replica)**: Tokens generated per second per replica
- **Decode Throughput (per Request)**: Tokens generated per second per individual request
- **Total System Throughput**: Total tokens per second across all replicas

#### Economic Analysis
- **Input Token Cost**: Cost per million input tokens
- **Output Token Cost**: Cost per million output tokens


## Technical Details

### Roofline Model
The calculator uses roofline models to estimate performance:
- **Prefill Phase**: Compute-bound, limited by GPU FLOPS
- **Decode Phase**: Memory-bound, limited by memory bandwidth

### MoE Calculations
For Mixture of Experts models:
- Only active expert parameters are used in calculations
- Active expert ratio = active experts / total experts
- Reduces computational requirements significantly

### Memory Calculations
- Model weights distributed across tensor parallel GPUs
- KV cache calculated per sequence length
- VRAM overhead accounts for activations and system memory


## License

This project is open source and available under the [MIT License](LICENSE).

## Support

For issues and questions:
- Open an issue on GitHub
- Check the [GitHub Discussions](https://github.com/yourusername/gpu-perf-calculator/discussions)

