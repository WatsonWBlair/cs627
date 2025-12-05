#!/bin/bash

# Docker Development Script for CS627 Semantic-Vector Space
# Launches Jupyter Lab development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    source .env
else
    source .env.docker
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CS627 Development Environment${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Detect GPU availability
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo -e "${GREEN}[OK] NVIDIA GPU detected and will be available${NC}"
        GPU_FLAGS="--gpus all"
    fi
fi

IMAGE="${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${DEV_TAG}"

# Check if image exists locally
if ! docker images | grep -q "${DOCKER_HUB_REPO}.*${DEV_TAG}"; then
    echo -e "${YELLOW}Development image not found locally.${NC}"
    echo -e "${YELLOW}Building development image...${NC}"
    docker build -f Dockerfile.dev -t ${IMAGE} .
fi

# Stop any existing dev container
if docker ps -a | grep -q cs627-dev; then
    echo -e "${YELLOW}Stopping existing dev container...${NC}"
    docker stop cs627-dev 2>/dev/null || true
    docker rm cs627-dev 2>/dev/null || true
fi

echo -e "\n${YELLOW}Starting development container...${NC}"

# Run development container
docker run \
    ${GPU_FLAGS} \
    --rm \
    -d \
    --name cs627-dev \
    -v $(pwd):/workspace \
    -p ${JUPYTER_PORT:-8888}:8888 \
    -p ${TENSORBOARD_PORT:-6006}:6006 \
    -e DEVICE=${DEVICE:-auto} \
    ${IMAGE}

# Wait for Jupyter to start
echo -e "${YELLOW}Waiting for Jupyter Lab to start...${NC}"
sleep 5

# Get the Jupyter URL
JUPYTER_URL="http://localhost:${JUPYTER_PORT:-8888}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Development Environment Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n${BLUE}Jupyter Lab:${NC} ${JUPYTER_URL}"
echo -e "${BLUE}TensorBoard:${NC} http://localhost:${TENSORBOARD_PORT:-6006}"
echo -e "\n${YELLOW}To access the container shell:${NC}"
echo -e "  docker exec -it cs627-dev bash"
echo -e "\n${YELLOW}To stop the container:${NC}"
echo -e "  docker stop cs627-dev"
echo -e "\n${YELLOW}To view logs:${NC}"
echo -e "  docker logs -f cs627-dev"

# Try to open browser (works on macOS and some Linux distributions)
if command -v open &> /dev/null; then
    echo -e "\n${YELLOW}Opening Jupyter Lab in browser...${NC}"
    open ${JUPYTER_URL}
elif command -v xdg-open &> /dev/null; then
    echo -e "\n${YELLOW}Opening Jupyter Lab in browser...${NC}"
    xdg-open ${JUPYTER_URL}
else
    echo -e "\n${YELLOW}Please open your browser and navigate to:${NC}"
    echo -e "  ${BLUE}${JUPYTER_URL}${NC}"
fi

# Show container logs
echo -e "\n${YELLOW}Container logs:${NC}"
docker logs cs627-dev