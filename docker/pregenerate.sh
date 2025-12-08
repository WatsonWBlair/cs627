#!/bin/bash

# CS627 Token Pre-generation Script
# Generates encoder tokens for adapter training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}CS627 Token Pre-generation${NC}"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Auto-detect GPU availability
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected, using GPU mode${NC}"
    GPU_FLAG="--gpus all"
    IMAGE_TAG="gpu"
    DEVICE="cuda"
else
    echo -e "${YELLOW}No GPU detected, using CPU mode${NC}"
    GPU_FLAG=""
    IMAGE_TAG="cpu"
    DEVICE="cpu"
fi

# Parse command line arguments
BATCH_SIZE=${BATCH_SIZE:-32}
USE_COMPOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
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
            echo "  --batch-size SIZE  Batch size for token generation (default: 32)"
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
echo "Creating data directories..."
mkdir -p data/cmumosi/{mosi,audio,frames}
mkdir -p data/pregenerated_tokens/mosi
mkdir -p OptimalWeights

if [ "$USE_COMPOSE" = true ]; then
    echo -e "${GREEN}Using docker-compose to run token pre-generation${NC}"
    docker-compose up pregenerate-tokens
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
    
    # Run token pre-generation
    echo -e "${GREEN}Starting token pre-generation...${NC}"
    echo "Configuration:"
    echo "  Device: ${DEVICE}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Output: data/pregenerated_tokens/mosi/"
    
    docker run \
        ${GPU_FLAG} \
        --rm \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/src:/workspace/src \
        -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
        -e DEVICE=${DEVICE} \
        -e BATCH_SIZE=${BATCH_SIZE} \
        -e OUTPUT_DIR=/workspace/data/pregenerated_tokens \
        -e MOSI_DATA_PATH=/workspace/data/cmumosi/mosi/ \
        -e AUDIO_DIR=/workspace/data/cmumosi/audio/ \
        -e VIDEO_DIR=/workspace/data/cmumosi/frames/ \
        ${IMAGE_NAME} \
        python src/Training/pregenerate_tokens.py
fi

# Check if tokens were generated
if [ -f "data/pregenerated_tokens/mosi/train_tokens.h5" ]; then
    echo -e "${GREEN}✓ Token pre-generation complete!${NC}"
    echo "Tokens saved to: data/pregenerated_tokens/mosi/"
    
    # Show file sizes
    echo ""
    echo "Generated files:"
    ls -lh data/pregenerated_tokens/mosi/*.h5 2>/dev/null || true
else
    echo -e "${RED}✗ Token generation may have failed${NC}"
    echo "Check the logs above for errors"
    exit 1
fi