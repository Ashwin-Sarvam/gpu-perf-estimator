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

## Hosting on GitHub Pages

### Method 1: Direct GitHub Pages (Recommended)

1. **Create a GitHub Repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/gpu-perf-calculator.git
   git push -u origin main
   ```

2. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Click "Settings" tab
   - Scroll down to "Pages" section
   - Under "Source", select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Click "Save"

3. **Access Your Site**:
   - Your site will be available at: `https://yourusername.github.io/gpu-perf-calculator/`
   - GitHub will build and deploy automatically on every push

### Method 2: Using GitHub Actions (Advanced)

1. **Create `.github/workflows/deploy.yml`**:
   ```yaml
   name: Deploy to GitHub Pages
   
   on:
     push:
       branches: [ main ]
   
   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Deploy to GitHub Pages
           uses: peaceiris/actions-gh-pages@v3
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: ./
   ```

2. **Enable GitHub Actions**:
   - Go to repository Settings → Actions → General
   - Enable "Allow GitHub Actions to create and approve pull requests"

### Method 3: Using a Custom Domain

1. **Add a CNAME file**:
   ```bash
   echo "yourdomain.com" > CNAME
   git add CNAME
   git commit -m "Add custom domain"
   git push
   ```

2. **Configure DNS**:
   - Add a CNAME record pointing to `yourusername.github.io`
   - Wait for DNS propagation (up to 24 hours)

## File Structure

```
gpu-perf-calculator/
├── index.html          # Main HTML structure
├── style.css           # Styling and layout
├── script.js           # JavaScript logic and calculations
├── README.md           # This file
└── .github/
    └── workflows/
        └── deploy.yml  # GitHub Actions workflow (optional)
```

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

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

For issues and questions:
- Open an issue on GitHub
- Check the [GitHub Discussions](https://github.com/yourusername/gpu-perf-calculator/discussions)

## Acknowledgments

- Based on roofline models for GPU performance analysis
- Inspired by SGLang's performance benchmarks
- Uses Hugging Face model configurations# gpu-perf-estimator
