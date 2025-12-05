#!/bin/bash

# Docker Training Script for CS627 Semantic-Vector Space
# Automatically detects GPU availability and runs appropriate container
#
# Usage:
#   ./docker/train.sh              # Train encoders (default)
#   ./docker/train.sh --decoders   # Train decoders
#   ./docker/train.sh -d           # Train decoders (short form)
#   ./docker/train.sh --compose    # Use docker-compose
#   ./docker/train.sh --gpu        # Force GPU mode

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments for training type
TRAIN_DECODERS=false
TRAINING_SCRIPT="src/Training/train_encoders.py"
TRAINING_TYPE="encoders"

for arg in "$@"; do
    case $arg in
        --decoders|-d)
            TRAIN_DECODERS=true
            TRAINING_SCRIPT="src/Training/train_decoders.py"
            TRAINING_TYPE="decoders"
            shift
            ;;
    esac
done

# Load environment variables
if [ -f .env ]; then
    source .env
else
    source .env.docker
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CS627 Docker Training Script${NC}"
echo -e "${GREEN}Training: ${TRAINING_TYPE}${NC}"
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
        echo -e "${GREEN}[OK] NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv
    fi
fi

# Determine which image and service to use
if [ "$GPU_AVAILABLE" = true ] || [[ "$*" == *"--gpu"* ]]; then
    IMAGE="${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${GPU_TAG}"
    if [ "$TRAIN_DECODERS" = true ]; then
        SERVICE="train-decoders-gpu"
    else
        SERVICE="train-gpu"
    fi
    echo -e "${YELLOW}Using GPU-enabled image${NC}"
    GPU_FLAGS="--gpus all"
else
    IMAGE="${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${CPU_TAG}"
    if [ "$TRAIN_DECODERS" = true ]; then
        SERVICE="train-decoders-cpu"
    else
        SERVICE="train-cpu"
    fi
    echo -e "${YELLOW}Using CPU-only image${NC}"
    GPU_FLAGS=""
fi

# Check if using docker-compose or standalone
if [[ "$*" == *"--compose"* ]]; then
    echo -e "\n${YELLOW}Starting training with docker-compose...${NC}"
    docker-compose up ${SERVICE}
else
    echo -e "\n${YELLOW}Starting standalone training container...${NC}"

    # Create necessary directories if they don't exist
    mkdir -p data/cmumosi/{mosi,audio,frames,videos}
    mkdir -p OptimalWeights CandidateWeights training_reports
    mkdir -p EncoderCheckpoints DecoderCheckpoints

    # Build environment variables for decoder training
    EXTRA_ENV=""
    if [ "$TRAIN_DECODERS" = true ]; then
        EXTRA_ENV="-e TRAIN_TEXT=${TRAIN_TEXT:-1} -e TRAIN_AUDIO=${TRAIN_AUDIO:-0} -e TRAIN_IMAGE=${TRAIN_IMAGE:-0} -e SEMANTIC_SOURCE=${SEMANTIC_SOURCE:-text_only}"
    fi

    # Run container
    docker run \
        ${GPU_FLAGS} \
        --rm \
        -it \
        --name cs627-training-${TRAINING_TYPE} \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
        -v $(pwd)/CandidateWeights:/workspace/CandidateWeights \
        -v $(pwd)/training_reports:/workspace/training_reports \
        -v $(pwd)/EncoderCheckpoints:/workspace/EncoderCheckpoints \
        -v $(pwd)/DecoderCheckpoints:/workspace/DecoderCheckpoints \
        -e DEVICE=${DEVICE:-auto} \
        -e BATCH_SIZE=${BATCH_SIZE:-32} \
        -e LEARNING_RATE=${LEARNING_RATE:-0.0001} \
        -e EPOCHS=${EPOCHS:-30} \
        -e INSTANCE_ID=${HOSTNAME:-docker} \
        ${EXTRA_ENV} \
        ${IMAGE} \
        python ${TRAINING_SCRIPT}
fi

echo -e "\n${GREEN}Training complete!${NC}"
echo -e "Check results in:"
echo -e "  - ${YELLOW}training_reports/${NC} for metrics and plots"
echo -e "  - ${YELLOW}CandidateWeights/${NC} for trained model weights"