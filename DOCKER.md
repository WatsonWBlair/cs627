# Docker Documentation for CS627 Semantic-Vector Space

## Overview

This project provides Docker containers for the CS627 Semantic-Vector Space multimodal training environment. The containers are optimized for both local development and cloud deployment, with support for GPU acceleration and CPU-only execution.

## Available Images

All images are available on Docker Hub at: `watsonwb/cs627-svs`

| Tag | Description | Use Case | Size |
|-----|-------------|----------|------|
| `latest` / `gpu` | GPU-enabled training image | Production training with NVIDIA GPUs | ~8GB |
| `cpu` | CPU-only training image | Testing, CI/CD, or CPU-only systems | ~4GB |
| `dev` | Development image with Jupyter Lab | Interactive development and experimentation | ~9GB |

## Quick Start

### Prerequisites

- Docker installed (version 20.10 or later)
- Docker Compose (optional, for multi-container setup)
- NVIDIA Docker runtime (for GPU support)
- At least 16GB RAM recommended
- 50GB free disk space for data and models

### Pull and Run (Recommended)

```bash
# Pull the latest GPU image
docker pull watsonwb/cs627-svs:latest

# Step 1: Generate tokens (run once)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/src:/workspace/src \
  -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
  watsonwb/cs627-svs:latest \
  python src/Training/pregenerate_tokens.py

# Step 2: Train adapters (fast training)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/src:/workspace/src \
  -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
  -v $(pwd)/Results:/workspace/Results \
  watsonwb/cs627-svs:latest \
  python src/Training/train_adapters.py --mode encoder

# CPU version
docker run \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/src:/workspace/src \
  -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
  watsonwb/cs627-svs:cpu \
  python src/Training/train_adapters.py --mode encoder

# Launch development environment
docker run --gpus all -p 8888:8888 \
  -v $(pwd):/workspace \
  watsonwb/cs627-svs:dev
```

## Using Helper Scripts

The `docker/` directory contains helper scripts for common operations:

### Building Images

```bash
# Build all images locally
./docker/build.sh

# Images will be tagged as:
# - watsonwb/cs627-svs:gpu
# - watsonwb/cs627-svs:cpu
# - watsonwb/cs627-svs:dev
# - watsonwb/cs627-svs:latest (same as gpu)
```

### Token-Based Training Workflow

```bash
# Step 1: Pre-generate tokens (run once)
./docker/pregenerate.sh

# Step 2: Train adapters (fast)
./docker/train-adapters.sh

# Options for both scripts:
# --batch-size SIZE  Set batch size
# --cpu              Force CPU mode
# --gpu              Force GPU mode
# --compose          Use docker-compose

# Train decoder adapters
./docker/train-adapters.sh --mode decoder

# Run ablation study
docker-compose up ablation-study
```

### Development Environment

```bash
# Start Jupyter Lab on http://localhost:8888
./docker/dev.sh

# Access container shell
docker exec -it cs627-dev bash

# Stop development container
docker stop cs627-dev
```

### Docker Hub Operations

```bash
# Push images to Docker Hub (requires login)
./docker/push.sh

# Pull latest images from Docker Hub
./docker/pull.sh          # Pull all images
./docker/pull.sh --gpu     # Pull GPU image only
./docker/pull.sh --cpu     # Pull CPU image only
./docker/pull.sh --dev     # Pull development image only
```

## Docker Compose

Use Docker Compose for multi-container orchestration:

```bash
# Token-based training workflow
docker-compose up pregenerate-tokens      # Step 1: Generate tokens
docker-compose up train-adapters-gpu      # Step 2: Train adapters (GPU)
docker-compose up train-adapters-cpu      # Or train on CPU

# Run ablation study
docker-compose up ablation-study

# Start development environment
docker-compose up dev

# Run evaluation
docker-compose up evaluate

# Preprocess data
docker-compose up preprocess

# Run tests
docker-compose up test
```

## Volume Mounts

The containers use several volume mounts for data persistence:

| Container Path | Host Path | Description |
|---------------|-----------|-------------|
| `/workspace/data` | `./data` | Training datasets (MOSI, etc.) |
| `/workspace/OptimalWeights` | `./OptimalWeights` | Pretrained model weights |
| `/workspace/CandidateWeights` | `./CandidateWeights` | Training output weights |
| `/workspace/training_reports` | `./training_reports` | Training metrics and plots |

## Environment Variables

Configure training parameters via environment variables:

```bash
# Training configuration
BATCH_SIZE=32              # Batch size for training
LEARNING_RATE=0.0001      # Learning rate
EPOCHS=30                 # Number of training epochs
DEVICE=auto               # Device: cuda, cpu, or auto

# Data paths
MOSI_DATA_PATH=/workspace/data/cmumosi/mosi/
AUDIO_DIR=/workspace/data/cmumosi/audio/
VIDEO_DIR=/workspace/data/cmumosi/frames/

# Model weights
OPTIMAL_WEIGHTS_DIR=/workspace/OptimalWeights
CANDIDATE_WEIGHTS_DIR=/workspace/CandidateWeights
```

### Using .env File

Create a `.env` file from the template:

```bash
cp .env.docker .env
# Edit .env with your configuration
```

Then run with docker-compose which will automatically load the .env file.

## GPU Support

### NVIDIA GPU Requirements

- NVIDIA drivers (version 470.57.01 or later)
- NVIDIA Docker runtime
- CUDA-capable GPU (compute capability 3.5 or higher)

### Verify GPU Setup

```bash
# Check if GPU is available
nvidia-smi

# Test GPU in container
docker run --gpus all watsonwb/cs627-svs:gpu \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Multi-GPU Support

```bash
# Use all GPUs
docker run --gpus all ...

# Use specific GPUs
docker run --gpus '"device=0,1"' ...

# Use single GPU
docker run --gpus '"device=0"' ...
```

## Cloud Deployment

### AWS EC2

```bash
# Launch g5.4xlarge with Deep Learning AMI
# Then:
ssh ubuntu@<instance-ip>
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
./docker/pull.sh --gpu
./docker/train.sh
```

### Google Cloud Platform

```bash
# Create instance with GPU
gcloud compute instances create cs627-training \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release

# SSH and run
gcloud compute ssh cs627-training
# ... same as AWS
```

### Azure

```bash
# Create GPU VM
az vm create \
  --resource-group myResourceGroup \
  --name cs627-vm \
  --image microsoft-dsvm:ubuntu-20-04 \
  --size Standard_NC6 \
  --admin-username azureuser

# SSH and run
ssh azureuser@<vm-ip>
# ... same as AWS
```

## Troubleshooting

### Out of Memory Errors

- Reduce batch size: `BATCH_SIZE=16` or `BATCH_SIZE=8`
- Use CPU version for testing
- Increase Docker memory limit in Docker Desktop settings

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# If fails, install NVIDIA Docker:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Slow Performance

- Ensure Docker has enough resources (CPU, Memory) in Docker Desktop settings
- Use SSD for data storage
- Consider using the CPU version for development

## Building Custom Images

### Modify for Your Needs

1. Edit Dockerfiles as needed
2. Build with custom tag:

```bash
docker build -f Dockerfile -t myuser/cs627-custom:latest .
```

3. Push to your registry:

```bash
docker push myuser/cs627-custom:latest
```

### Multi-Architecture Builds

```bash
# Setup buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t watsonwb/cs627-svs:cpu \
  -f Dockerfile.cpu \
  --push .
```

## CI/CD Integration

### GitHub Actions

The project includes GitHub Actions workflow for automated Docker builds:

- Triggers on push to main branch
- Builds and pushes all images to Docker Hub
- Runs basic tests on each image

See `.github/workflows/docker-publish.yml` for configuration.

### Manual Trigger

```bash
# Trigger workflow manually from GitHub UI
# Actions -> Docker Image CI/CD -> Run workflow
```

## Security Considerations

- Never commit `.env` files with secrets
- Use Docker secrets for sensitive data in production
- Regularly update base images for security patches
- Scan images for vulnerabilities:

```bash
docker scan watsonwb/cs627-svs:latest
```

## License

MIT License - See LICENSE file for details.

All Docker images include the MIT license and are free for educational and research use.

## Support

- GitHub Issues: https://github.com/WatsonWBlair/cs627/issues
- Docker Hub: https://hub.docker.com/r/watsonwb/cs627-svs

## Contributing

1. Fork the repository
2. Create your feature branch
3. Test your Docker changes locally
4. Submit a pull request

See CONTRIBUTING.md for detailed guidelines.