# Docker Guide

Docker setup for CS627 Semantic-Vector Space project.

## Quick Start

### Pull and Run

```bash
# Pull image
docker pull watsonwb/cs627-svs:latest

# Generate tokens
docker run --gpus all -v $(pwd)/data:/workspace/data \
  watsonwb/cs627-svs:latest \
  python src/Training/pregenerate_tokens.py

# Train adapters
docker run --gpus all -v $(pwd)/data:/workspace/data \
  -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
  watsonwb/cs627-svs:latest \
  python src/Training/train_adapters.py
```

## Available Images

| Image | Description | Size |
|-------|-------------|------|
| `watsonwb/cs627-svs:latest` | GPU-enabled (default) | 8GB |
| `watsonwb/cs627-svs:gpu` | GPU-enabled | 8GB |
| `watsonwb/cs627-svs:cpu` | CPU-only | 4GB |
| `watsonwb/cs627-svs:dev` | Jupyter development | 9GB |

## Using Docker Compose

### Basic Workflow

```bash
# 1. Generate tokens (one-time)
docker-compose up pregenerate-tokens

# 2. Train adapters
docker-compose up train-adapters-gpu

# 3. Run evaluation
docker-compose up evaluate

# 4. Ablation study
docker-compose up ablation-study
```

### Development Environment

```bash
# Launch Jupyter Lab
docker-compose up dev
# Access at http://localhost:8888
```

## Using Helper Scripts

### Setup
```bash
# Build all images locally
./docker/build.sh

# Pull from Docker Hub
./docker/pull.sh
```

### Training
```bash
# Generate tokens
./docker/pregenerate.sh

# Train adapters
./docker/train-adapters.sh

# Options
./docker/train-adapters.sh --mode decoder
./docker/train-adapters.sh --batch-size 512
./docker/train-adapters.sh --cpu
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/workspace/data` | Datasets and tokens |
| `./src` | `/workspace/src` | Source code |
| `./OptimalWeights` | `/workspace/OptimalWeights` | Trained models |
| `./Results` | `/workspace/Results` | Training logs |

## GPU Support

### Check GPU
```bash
# Test GPU access
docker run --rm --gpus all watsonwb/cs627-svs:gpu nvidia-smi
```

### Requirements
- NVIDIA GPU with 8GB+ VRAM
- NVIDIA Docker runtime
- CUDA 11.8 or 12.1

### CPU Fallback
```bash
# Use CPU image if no GPU
docker run -v $(pwd)/data:/workspace/data \
  watsonwb/cs627-svs:cpu \
  python src/Training/train_adapters.py
```

## Building Images

### Build Locally
```bash
# GPU image
docker build -t watsonwb/cs627-svs:gpu .

# CPU image
docker build -f Dockerfile.cpu -t watsonwb/cs627-svs:cpu .

# Dev image
docker build -f Dockerfile.dev -t watsonwb/cs627-svs:dev .
```

### Push to Docker Hub
```bash
# Login
docker login -u watsonwb

# Push images
docker push watsonwb/cs627-svs:gpu
docker push watsonwb/cs627-svs:cpu
docker push watsonwb/cs627-svs:latest
```

## Environment Variables

### Default Configuration
```bash
DEVICE=cuda                    # cuda or cpu
BATCH_SIZE=32                  # Token generation
ADAPTER_BATCH_SIZE=256         # Training
EPOCHS=50
LEARNING_RATE=0.001
```

### Custom Settings
```bash
# Override defaults
docker run --gpus all \
  -e BATCH_SIZE=64 \
  -e EPOCHS=100 \
  -e LEARNING_RATE=0.0005 \
  -v $(pwd)/data:/workspace/data \
  watsonwb/cs627-svs:latest \
  python src/Training/train_adapters.py
```

## Common Commands

### Interactive Shell
```bash
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  watsonwb/cs627-svs:latest \
  bash
```

### Run Tests
```bash
docker run --rm watsonwb/cs627-svs:latest \
  python -m pytest tests/unit/smoke_test.py
```

### Clean Up
```bash
# Remove containers
docker-compose down

# Remove images
docker rmi watsonwb/cs627-svs:latest

# Clean everything
docker system prune -a
```

## Troubleshooting

### "no such file or directory"
```bash
# Create required directories
mkdir -p data/cmumosi/{mosi,audio,frames}
mkdir -p data/pregenerated_tokens/mosi
mkdir -p OptimalWeights Results
```

### "permission denied"
```bash
# Fix permissions (Linux)
sudo chown -R $USER:$USER data OptimalWeights Results
```

### "out of memory"
```bash
# Reduce batch size
docker run --gpus all \
  -e BATCH_SIZE=8 \
  -v $(pwd)/data:/workspace/data \
  watsonwb/cs627-svs:latest \
  python src/Training/pregenerate_tokens.py
```

### "GPU not available"
```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Next Steps

- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Training workflows
- [AWS_SETUP.md](AWS_SETUP.md) - Cloud deployment
- [SETUP.md](SETUP.md) - Local installation