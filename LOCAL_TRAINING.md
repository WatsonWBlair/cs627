# Local Training Guide

Step-by-step instructions for running token pregeneration and adapter training locally on macOS, Windows, and Linux.

## Overview

The training workflow has two phases:

1. **Pre-generate Tokens** (~30 min): Process raw data through frozen encoders once
2. **Train Adapters** (~15 min): Train lightweight MLPs on pre-generated tokens

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for advanced options and [SETUP.md](SETUP.md) for detailed prerequisites.

## Prerequisites

| Requirement | Minimum |
|-------------|---------|
| RAM | 16GB |
| Storage | 50GB free |
| GPU | Optional (8GB+ VRAM recommended) |
| Python | 3.8+ |

## Platform Setup

### macOS

```bash
# Install Python (if needed)
brew install python@3.10

# Clone repository
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU)
pip install torch torchvision torchaudio

# Or for Apple Silicon (M1/M2/M3)
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt

# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..
```

**Docker alternative:**
```bash
brew install --cask docker
open -a Docker
make docker-build && make tokens && make train
```

### Windows

**Option A: Native Windows (PowerShell/CMD)**

```powershell
# Clone repository
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch (CPU)
pip install torch torchvision torchaudio

# Or with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK
pip install .
cd ..
```

**Option B: WSL2 (Recommended for GPU)**

```bash
# In WSL2 Ubuntu terminal
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..
```

**Docker alternative (requires NVIDIA GPU + drivers):**
```powershell
make.bat docker-build
make.bat tokens        # GPU Docker
make.bat train         # GPU Docker
```

**Docker CPU (no GPU required):**
```powershell
make.bat tokens-cpu    # CPU Docker
make.bat train-cpu     # CPU Docker
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3.10 python3-pip python3-venv git

# Clone repository
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..
```

**Docker alternative:**
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
make docker-build && make tokens && make train
```

## Training Workflow

### Step 1: Create Directories

```bash
mkdir -p data/cmumosi/{mosi,audio,frames}
mkdir -p data/pregenerated_tokens/mosi
mkdir -p OptimalWeights Results
```

### Step 2: Download Dataset

```bash
# Download CMU-MOSI data
SKIP_DOWNLOAD=0 python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"
```

### Step 3: Pre-generate Tokens

```bash
python src/Training/pregenerate_tokens.py
```

Adjust batch size for your hardware:
```bash
# Limited memory (8GB RAM/VRAM)
BATCH_SIZE=8 python src/Training/pregenerate_tokens.py

# Windows CMD (different env var syntax)
set BATCH_SIZE=8
python src/Training/pregenerate_tokens.py
```

**Expected output** in `data/pregenerated_tokens/mosi/`:
- `train_tokens.h5` (~1.5GB)
- `val_tokens.h5` (~300MB)
- `test_tokens.h5` (~300MB)
- `metadata.json`

### Step 4: Train Encoder Adapters

```bash
python src/Training/train_adapters.py --mode encoder
```

Custom parameters:
```bash
python src/Training/train_adapters.py --mode encoder --epochs 100 --batch-size 512 --lr 0.0005
```

### Step 5: Train Decoder Adapters

```bash
python src/Training/train_adapters.py --mode decoder
```

### Step 6: Verify Results

Check that weights were saved:
```bash
ls OptimalWeights/
# Expected: text_adapter_weights.pth, audio_adapter_weights.pth, etc.
```

## Ablation Studies

Run ablation studies to compare different adapter configurations:

```bash
# Run recommended configurations
python src/Experiments/run_ablation.py --config recommended

# Run full grid search (slower, more comprehensive)
python src/Experiments/run_ablation.py --config grid_search

# Windows make.bat
make.bat ablation-local
```

### Visualize Results

Generate visualizations from ablation results:

```bash
# Visualize latest results
python scripts/visualize_ablation.py

# Specify results file
python scripts/visualize_ablation.py --results Results/ablation/ablation_results_*.json

# Windows make.bat
make.bat visualize-ablation
```

**Output figures** in `Results/ablation/figures/`:
- `config_comparison.png` - Bar chart comparing validation loss
- `training_curves.png` - Training/validation loss curves
- `parameter_heatmap.png` - Hyperparameter effect analysis
- `best_config_summary.png` - Summary of best configuration

## Quick Reference

### Direct Python (Recommended)

| Task | Command |
|------|---------|
| Generate tokens | `python src/Training/pregenerate_tokens.py` |
| Train encoder adapters | `python src/Training/train_adapters.py --mode encoder` |
| Train decoder adapters | `python src/Training/train_adapters.py --mode decoder` |
| Run ablation study | `python src/Experiments/run_ablation.py --config recommended` |
| Visualize ablation | `python scripts/visualize_ablation.py` |
| Visualize performance | `python scripts/generate_performance_figures.py` |
| Visualize decoder | `python scripts/generate_decoder_figures.py` |

### Windows make.bat

| Task | Command |
|------|---------|
| Generate tokens (no Docker) | `make.bat tokens-local` |
| Generate tokens (CPU Docker) | `make.bat tokens-cpu` |
| Generate tokens (GPU Docker) | `make.bat tokens` |
| Train adapters (no Docker) | `make.bat train-local` |
| Train adapters (CPU Docker) | `make.bat train-cpu` |
| Train adapters (GPU Docker) | `make.bat train` |
| Run ablation study | `make.bat ablation-local` |
| Visualize ablation | `make.bat visualize-ablation` |
| Visualize performance | `make.bat visualize-performance` |
| Visualize decoder | `make.bat visualize-decoder` |

### Environment Variables

```bash
# Reduce batch size for limited memory
BATCH_SIZE=8 python src/Training/pregenerate_tokens.py

# Enable mixed precision
USE_AMP=1 python src/Training/train_adapters.py --mode encoder
```

**Windows CMD syntax:**
```cmd
set BATCH_SIZE=8
python src/Training/pregenerate_tokens.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'mmsdk'` | Install CMU-MultimodalSDK (see platform setup) |
| CUDA out of memory | Reduce `BATCH_SIZE` (8 or 16) |
| `make` not found (Windows) | Use `make.bat` or run Python commands directly |
| Slow training on CPU | Enable `USE_AMP=1` or use Docker with GPU |
| Token files not found | Run `pregenerate_tokens.py` first |
| `nvidia-container-cli: initialization error` | Use `make.bat tokens-local` or `make.bat tokens-cpu` instead |
| Docker GPU not working on Windows | Use `-local` commands or WSL2 with NVIDIA drivers |

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for extended troubleshooting.

## Next Steps

- [EVALUATION.md](EVALUATION.md) - Evaluate trained models
- [BENCHMARKS.md](BENCHMARKS.md) - Performance baselines
- [AWS_SETUP.md](AWS_SETUP.md) - Cloud training
