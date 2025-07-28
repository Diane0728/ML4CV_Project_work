#!/bin/bash
# Quick setup script for FastVLM Mobile Optimization Project

echo "🚀 FastVLM Mobile Optimization Project Setup"
echo "==========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Error: Python 3.10+ is required. Current version: $python_version"
    echo "Please install Python 3.10 or higher."
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d "fastvlm_env" ]; then
    python3 -m venv fastvlm_env
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source fastvlm_env/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install FastVLM
echo ""
echo "📦 Installing FastVLM..."
if [ -d "ml-fastvlm" ]; then
    cd ml-fastvlm
    pip install -e . --quiet
    cd ..
    echo "✅ FastVLM installed"
else
    echo "❌ Error: ml-fastvlm directory not found"
    echo "Please ensure you're in the ML4CV_FastVLM_Project directory"
    exit 1
fi

# Install additional requirements
echo ""
echo "📦 Installing additional dependencies..."
pip install matplotlib --quiet
echo "✅ Additional dependencies installed"

# Check if model exists
echo ""
echo "🤖 Checking model weights..."
if [ -d "ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3" ]; then
    echo "✅ Model weights found"
else
    echo "⚠️  Model weights not found. Downloading..."
    mkdir -p ml-fastvlm/checkpoints
    cd ml-fastvlm/checkpoints
    curl -L -O https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip
    unzip llava-fastvithd_0.5b_stage3.zip
    rm llava-fastvithd_0.5b_stage3.zip
    cd ../..
    echo "✅ Model weights downloaded"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "To run the experiments:"
echo "1. Activate the environment: source fastvlm_env/bin/activate"
echo "2. Run baseline benchmark: python src/benchmark_baseline.py"
echo "3. Generate analysis: python src/performance_analysis.py"
echo ""
echo "Check the README.md for detailed instructions."