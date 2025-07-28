# 🚀 Quick Start Guide for Colleagues

Welcome! This guide will help you run the FastVLM optimization experiments in under 5 minutes.

## Prerequisites Check ✅

You need:
- M1/M2/M3 Mac (Apple Silicon)
- Python 3.10+ installed
- 5GB free disk space

## Super Quick Setup (2 minutes)

1. **Open Terminal and navigate to project:**
   ```bash
   cd path/to/ML4CV_FastVLM_Project
   ```

2. **Run the automatic setup:**
   ```bash
   ./setup.sh
   ```
   
   This script will:
   - Check Python version
   - Create virtual environment
   - Install all dependencies
   - Download model if needed

## Run the Experiments (3 minutes)

1. **Activate the environment:**
   ```bash
   source fastvlm_env/bin/activate
   ```

2. **Run baseline benchmark:**
   ```bash
   python src/benchmark_baseline.py
   ```
   
   You'll see output like:
   ```
   Model loaded in 5.07 seconds
   Average tokens/second: 13.48
   ```

3. **Generate performance analysis:**
   ```bash
   python src/performance_analysis.py
   ```
   
   This creates a nice visualization in `results/performance_analysis.png`

## View Results 📊

1. **Check the performance graph:**
   ```bash
   open results/performance_analysis.png
   ```

2. **Read the summary:**
   ```bash
   cat results/optimization_summary.json | python -m json.tool
   ```

3. **Full project report:**
   ```bash
   open docs/PROJECT_REPORT.md
   ```

## What Each File Does 📁

- **src/benchmark_baseline.py**: Tests model speed with different prompts
- **src/performance_analysis.py**: Creates graphs and analysis
- **src/quantize_model.py**: Quantization experiments (experimental)
- **results/***: All experimental results stored here
- **docs/PROJECT_REPORT.md**: Full report for submission

## Common Issues & Fixes 🔧

**"No module named llava"**
```bash
cd ml-fastvlm && pip install -e . && cd ..
```

**"MPS backend out of memory"**
- Close Chrome/Safari
- Quit other heavy apps

**Slow performance**
- Make sure no other Python processes are running
- Check Activity Monitor → GPU tab

## Testing Your Own Images 🖼️

```bash
python src/benchmark_baseline.py
# It will use the test_image.jpg created automatically
# To use your own image, modify the image_path in the script
```

## Need Help? 🤝

1. Check the full README.md for detailed instructions
2. Look at the PROJECT_REPORT.md for technical details
3. The code is well-commented - check the source files

---

**Quick Commands Cheatsheet:**
```bash
# Setup (one time)
./setup.sh

# Every time you work
source fastvlm_env/bin/activate
python src/benchmark_baseline.py
python src/performance_analysis.py

# See results
open results/performance_analysis.png
```

That's it! The experiments should run in about 5 minutes total. 🎉