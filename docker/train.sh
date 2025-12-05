#!/bin/bash

# Docker Training Script for CS627 Semantic-Vector Space
# Automatically detects GPU availability and runs appropriate container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    source .env
else
    source .env.docker
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CS627 Docker Training Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Detect GPU availability
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv
    fi
fi

# Determine which image to use
if [ "$GPU_AVAILABLE" = true ] || [ "$1" = "--gpu" ]; then
    IMAGE="${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${GPU_TAG}"
    SERVICE="train-gpu"
    echo -e "${YELLOW}Using GPU-enabled image${NC}"
    GPU_FLAGS="--gpus all"
else
    IMAGE="${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${CPU_TAG}"
    SERVICE="train-cpu"
    echo -e "${YELLOW}Using CPU-only image${NC}"
    GPU_FLAGS=""
fi

# Check if using docker-compose or standalone
if [ "$2" = "--compose" ] || [ "$1" = "--compose" ]; then
    echo -e "\n${YELLOW}Starting training with docker-compose...${NC}"
    docker-compose up ${SERVICE}
else
    echo -e "\n${YELLOW}Starting standalone training container...${NC}"
    
    # Create necessary directories if they don't exist
    mkdir -p data/cmumosi/{mosi,audio,frames,videos}
    mkdir -p OptimalWeights CandidateWeights training_reports
    
    # Run container
    docker run \
        ${GPU_FLAGS} \
        --rm \
        -it \
        --name cs627-training \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
        -v $(pwd)/CandidateWeights:/workspace/CandidateWeights \
        -v $(pwd)/training_reports:/workspace/training_reports \
        -e DEVICE=${DEVICE:-auto} \
        -e BATCH_SIZE=${BATCH_SIZE:-32} \
        -e LEARNING_RATE=${LEARNING_RATE:-0.0001} \
        -e EPOCHS=${EPOCHS:-30} \
        -e INSTANCE_ID=${HOSTNAME:-docker} \
        ${IMAGE} \
        python src/Training/train_encoders.py
fi

echo -e "\n${GREEN}Training complete!${NC}"
echo -e "Check results in:"
echo -e "  - ${YELLOW}training_reports/${NC} for metrics and plots"
echo -e "  - ${YELLOW}CandidateWeights/${NC} for trained model weights"