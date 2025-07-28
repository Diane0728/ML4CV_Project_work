# ML4CV_Project_work
# FastVLM Mobile Optimization Project

> 🚀 Optimizing FastVLM-0.5B Vision-Language Model for Apple Silicon (M2)

This project explores performance optimization techniques for deploying state-of-the-art vision-language models on consumer hardware, specifically targeting the M2 MacBook Pro.

## 📁 Project Structure

```
ML4CV_FastVLM_Project/
├── src/                      # Source code
│   ├── benchmark_baseline.py # Baseline performance benchmarking
│   ├── quantize_model.py     # Quantization experiments
│   └── performance_analysis.py # Results analysis and visualization
├── results/                  # Experimental results
│   ├── benchmark_results_baseline.json
│   ├── quantization_results.json
│   ├── optimization_summary.json
│   └── performance_analysis.png
├── docs/                     # Documentation
│   └── PROJECT_REPORT.md     # Full project report
├── models/                   # Model checkpoints (created during setup)
├── ml-fastvlm/              # FastVLM repository (cloned during setup)
├── mlx-vlm/                 # MLX-VLM repository (cloned during setup)
└── test_image.jpg           # Test image for benchmarking
```

## 🛠️ Setup Instructions

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9 or higher
- At least 16GB RAM
- ~5GB free disk space

### Step 1: Clone the Project

```bash
cd ~/Laboratory/ML4CV\ project\ work/
cd ML4CV_FastVLM_Project
```

### Step 2: Create Python Environment

```bash
# Create virtual environment
python3 -m venv fastvlm_env

# Activate environment
source fastvlm_env/bin/activate
```

### Step 3: Install FastVLM

```bash
# Install FastVLM from the cloned repository
cd ml-fastvlm
pip install --upgrade pip
pip install -e .
cd ..
```

### Step 4: Download Model Weights

The FastVLM-0.5B model is already downloaded in `ml-fastvlm/checkpoints/`. If you need to re-download:

```bash
cd ml-fastvlm/checkpoints
curl -L -O https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip
unzip llava-fastvithd_0.5b_stage3.zip
rm llava-fastvithd_0.5b_stage3.zip
cd ../..
```

### Step 5: Install Additional Dependencies

```bash
pip install matplotlib
```

## 🚀 Running the Experiments

### 1. Baseline Benchmarking

Run the baseline performance test:

```bash
python src/benchmark_baseline.py
```

This will:
- Load FastVLM-0.5B model
- Create a test image (if not exists)
- Run 5 different prompts with 5 runs each
- Save results to `results/benchmark_results_baseline.json`

**Expected output:**
```
Starting FastVLM baseline benchmark on M2...
Model loaded in 5.07 seconds
Total parameters: 622.40M
Average tokens/second: 13.48
```

### 2. Quantization Experiments (Optional)

```bash
python src/quantize_model.py
```

Note: Standard PyTorch quantization is incompatible with FastVLM architecture, but this script demonstrates the approach.

### 3. Performance Analysis

Generate performance visualizations and summary:

```bash
python src/performance_analysis.py
```

This creates:
- `results/performance_analysis.png` - Performance visualization
- `results/optimization_summary.json` - Detailed analysis

## 📊 Understanding the Results

### Key Metrics

1. **Tokens per Second**: Main performance metric (baseline: ~13.48)
2. **Inference Time**: Time for 100-token generation (~4.1 seconds)
3. **Model Load Time**: Initial model loading (~5 seconds)
4. **Memory Usage**: ~1.2GB for model weights

### Performance Visualization

The `performance_analysis.png` shows:
- Token generation speed by prompt type
- Inference time distribution
- Model specifications
- Expected speedup with optimizations

## 🔧 Advanced: MLX Framework Setup (Optional)

For further optimization experiments with MLX:

```bash
# MLX-VLM is already cloned and patched
cd mlx-vlm
pip install -e .
cd ..

# Convert model (experimental)
python -m mlx_vlm.generate --model ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
                          --image test_image.jpg \
                          --prompt "Describe this image." \
                          --max-tokens 100
```

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'llava'"**
   - Make sure you're in the activated virtual environment
   - Reinstall: `cd ml-fastvlm && pip install -e .`

2. **"MPS backend out of memory"**
   - Close other applications
   - Reduce batch size in benchmarking scripts

3. **Slow performance**
   - Ensure you're using MPS (Metal Performance Shaders)
   - Check Activity Monitor for GPU usage

### Environment Check

```bash
# Check Python version
python --version  # Should be 3.10+

# Check PyTorch MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Check installed packages
pip list | grep -E "torch|transformers|llava"
```

## 📝 Project Report

The full project report is available in `docs/PROJECT_REPORT.md`. It includes:
- Detailed methodology
- Complete results analysis
- Technical challenges and solutions
- Future work recommendations

## 🤝 Contributing

To extend this project:

1. **Add new benchmarks**: Modify `src/benchmark_baseline.py`
2. **Try different models**: Change model path in scripts
3. **Test new optimizations**: Create new scripts in `src/`

## 📚 References

- [FastVLM Paper](https://arxiv.org/abs/2412.13303)
- [FastVLM GitHub](https://github.com/apple/ml-fastvlm)
- [MLX Framework](https://github.com/ml-explore/mlx)

## 📧 Contact

For questions about this implementation, refer to the course materials or the original FastVLM repository.

---
**Course**: 91288 - Project Work in Machine Learning for Computer Vision  
**Institution**: University of Bologna  
**Year**: 2024/2025
