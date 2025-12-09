# Setup Guide

Complete setup instructions for the CS627 Semantic-Vector Space project.

## System Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 50GB free space
- **GPU**: Optional but recommended (NVIDIA with 8GB+ VRAM)
- **OS**: macOS 12+, Windows 10/11 (WSL2), Ubuntu 20.04+

### Software Prerequisites
- Python 3.8+
- Docker 20.10+ (recommended)
- Git
- 10GB+ internet bandwidth for model downloads

## Platform-Specific Setup

### macOS

#### 1. Install Prerequisites
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python@3.10 git

# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop
open -a Docker
```

#### 2. Clone Repository
```bash
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
```

#### 3. Run Setup
```bash
# Quick automated setup
./quickstart.sh

# Or manual setup with invoke
pip install invoke
inv setup
inv tokens
inv train
```

### Windows

#### 1. Install Prerequisites

**Option A: WSL2 (Recommended)**
```powershell
# Enable WSL2 (run as Administrator)
wsl --install

# Install Ubuntu
wsl --install -d Ubuntu-22.04

# Restart computer, then open Ubuntu terminal
```

**Option B: Native Windows**
- Install [Python 3.10](https://www.python.org/downloads/)
- Install [Git](https://git-scm.com/download/win)
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

#### 2. Clone Repository
```bash
# In WSL2/Ubuntu terminal or Git Bash
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
```

#### 3. Run Setup
```bash
# All platforms (Windows, macOS, Linux)
pip install invoke
inv setup
inv tokens
inv train
```

### Linux (Ubuntu/Debian)

#### 1. Install Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.10 python3-pip git curl

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA drivers (if GPU available)
sudo apt install -y nvidia-driver-525
sudo apt install -y nvidia-docker2
```

#### 2. Clone Repository
```bash
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
```

#### 3. Run Setup
```bash
pip install invoke
inv setup
inv tokens
inv train
```

## Installation Methods

### Method 1: Docker (Recommended)

```bash
# Pull pre-built image
docker pull watsonwb/cs627-svs:latest

# Or build locally
inv docker-build

# Run training pipeline
inv tokens    # Generate tokens (30 min)
inv train     # Train adapters (15 min)
inv evaluate  # Run evaluation
```

### Method 2: Local Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..

# Create directories
mkdir -p data/cmumosi/{mosi,audio,frames}
mkdir -p data/pregenerated_tokens/mosi
mkdir -p OptimalWeights Results

# Download MOSI dataset
python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"
```

### Method 3: Quick Demo

```bash
# Run automated setup with demo data
./quickstart.sh
```

## Data Preparation

### Download CMU-MOSI Dataset

```bash
# Automatic download
inv download-data

# Or manual download
python scripts/data_wrangling/extract_test_segments.py
```

### Pre-generate Tokens

```bash
# Generate all tokens (~30 minutes on GPU)
inv tokens

# Or with custom batch size
BATCH_SIZE=16 python src/Training/pregenerate_tokens.py
```

Token files will be created in `data/pregenerated_tokens/mosi/`:
- `train_tokens.h5` (~1.5GB)
- `val_tokens.h5` (~300MB)
- `test_tokens.h5` (~300MB)

## Verification

### Test Installation
```bash
# Run smoke tests
inv test

# Or run Python test
python -c "
import torch
from src.Encoders import Text_to_Vec
encoder = Text_to_Vec()
result = encoder(['test'])
print(f'Success! Output shape: {result.shape}')
"
```

### Check GPU
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Docker GPU
docker run --rm --gpus all watsonwb/cs627-svs:gpu nvidia-smi
```

## Troubleshooting

### Common Issues

#### "No module named 'mmsdk'"
```bash
# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install .
```

#### "CUDA out of memory"
```bash
# Reduce batch size
BATCH_SIZE=8 inv tokens
BATCH_SIZE=64 inv train

# Or use local training with custom batch size
BATCH_SIZE=8 python src/Training/pregenerate_tokens.py
```

#### "Docker: permission denied"
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

#### PyTorch Installation Issues
```bash
# CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Environment Variables

See [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) for the complete reference.

Key variables for setup:
```bash
DEVICE=cuda                    # cuda or cpu
BATCH_SIZE=32                  # Token generation batch size
MOSI_DATA_PATH=data/cmumosi/mosi/
```

## Next Steps

1. **Generate tokens**: `inv tokens` (one-time, ~30 min)
2. **Train adapters**: `inv train` (fast, ~15 min)
3. **Run evaluation**: `inv evaluate`
4. **Ablation studies**: `inv ablation --local`

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training instructions.