#!/bin/bash

# CS627 Adapter Training Script
# Trains adapter MLPs using pre-generated tokens

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}CS627 Adapter Training${NC}"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if tokens exist
if [ ! -f "data/pregenerated_tokens/mosi/train_tokens.h5" ]; then
    echo -e "${RED}Error: No pre-generated tokens found${NC}"
    echo "Please run ./docker/pregenerate.sh first"
    exit 1
fi

# Auto-detect GPU availability
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected, using GPU mode${NC}"
    GPU_FLAG="--gpus all"
    IMAGE_TAG="gpu"
    DEVICE="cuda"
    DEFAULT_BATCH_SIZE=256
else
    echo -e "${YELLOW}No GPU detected, using CPU mode${NC}"
    GPU_FLAG=""
    IMAGE_TAG="cpu"
    DEVICE="cpu"
    DEFAULT_BATCH_SIZE=128
fi

# Parse command line arguments
MODE="encoder"
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
EPOCHS=${EPOCHS:-50}
LR=${LR:-0.001}
USE_COMPOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --cpu)
            echo "Forcing CPU mode"
            GPU_FLAG=""
            IMAGE_TAG="cpu"
            DEVICE="cpu"
            shift
            ;;
        --gpu)
            echo "Forcing GPU mode"
            GPU_FLAG="--gpus all"
            IMAGE_TAG="gpu"
            DEVICE="cuda"
            shift
            ;;
        --compose)
            USE_COMPOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --mode MODE       Training mode: encoder or decoder (default: encoder)"
            echo "  --batch-size SIZE Batch size for training (default: 256 for GPU, 128 for CPU)"
            echo "  --epochs N        Number of epochs (default: 50)"
            echo "  --lr RATE         Learning rate (default: 0.001)"
            echo "  --cpu             Force CPU mode"
            echo "  --gpu             Force GPU mode"
            echo "  --compose         Use docker-compose"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create necessary directories
echo "Creating directories..."
mkdir -p OptimalWeights
mkdir -p Results/adapter_training

if [ "$USE_COMPOSE" = true ]; then
    echo -e "${GREEN}Using docker-compose to run adapter training${NC}"
    
    if [ "$DEVICE" = "cuda" ]; then
        SERVICE="train-adapters-gpu"
    else
        SERVICE="train-adapters-cpu"
    fi
    
    BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS LEARNING_RATE=$LR \
        docker-compose up ${SERVICE}
else
    # Pull or build image
    IMAGE_NAME="watsonblair/cs627-svs:${IMAGE_TAG}"
    
    echo "Using image: ${IMAGE_NAME}"
    
    # Check if image exists locally
    if ! docker image inspect ${IMAGE_NAME} &> /dev/null; then
        echo -e "${YELLOW}Image not found locally, pulling from Docker Hub...${NC}"
        docker pull ${IMAGE_NAME}
        
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Pull failed, building locally...${NC}"
            docker build -f Dockerfile -t ${IMAGE_NAME} .
        fi
    fi
    
    # Run adapter training
    echo -e "${GREEN}Starting adapter training...${NC}"
    echo "Configuration:"
    echo "  Mode: ${MODE}"
    echo "  Device: ${DEVICE}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Epochs: ${EPOCHS}"
    echo "  Learning rate: ${LR}"
    echo ""
    
    docker run \
        ${GPU_FLAG} \
        --rm \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/src:/workspace/src \
        -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
        -v $(pwd)/Results:/workspace/Results \
        -e DEVICE=${DEVICE} \
        -e BATCH_SIZE=${BATCH_SIZE} \
        -e LEARNING_RATE=${LR} \
        -e EPOCHS=${EPOCHS} \
        -e TOKEN_DIR=/workspace/data/pregenerated_tokens/mosi \
        -e USE_AMP=$([ "$DEVICE" = "cuda" ] && echo "1" || echo "0") \
        ${IMAGE_NAME} \
        python src/Training/train_adapters.py --mode ${MODE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR}
fi

# Check if training completed
if [ -d "Results/adapter_training" ] && [ "$(ls -A Results/adapter_training)" ]; then
    echo -e "${GREEN}✓ Adapter training complete!${NC}"
    echo "Results saved to: Results/adapter_training/"
    
    # Show latest results
    echo ""
    echo "Latest training results:"
    ls -lt Results/adapter_training/ | head -5
else
    echo -e "${YELLOW}Training completed, check logs for details${NC}"
fi