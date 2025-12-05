#!/bin/bash

# Docker Pull Script for CS627 Semantic-Vector Space
# Pulls latest images from Docker Hub

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
echo -e "${GREEN}CS627 Docker Pull Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Function to pull an image
pull_image() {
    local tag=$1
    local description=$2
    
    echo -e "\n${YELLOW}Pulling ${description}...${NC}"
    echo -e "Tag: ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${tag}"
    
    docker pull ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${tag}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${description} pulled successfully${NC}"
    else
        echo -e "${RED}✗ Failed to pull ${description}${NC}"
        return 1
    fi
}

# Parse arguments
PULL_ALL=false
PULL_GPU=false
PULL_CPU=false
PULL_DEV=false

if [ $# -eq 0 ] || [ "$1" = "--all" ]; then
    PULL_ALL=true
elif [ "$1" = "--gpu" ]; then
    PULL_GPU=true
elif [ "$1" = "--cpu" ]; then
    PULL_CPU=true
elif [ "$1" = "--dev" ]; then
    PULL_DEV=true
else
    echo "Usage: $0 [--all|--gpu|--cpu|--dev]"
    echo "  --all  Pull all images (default)"
    echo "  --gpu  Pull GPU image only"
    echo "  --cpu  Pull CPU image only"
    echo "  --dev  Pull development image only"
    exit 1
fi

# Pull requested images
if [ "$PULL_ALL" = true ]; then
    pull_image "${GPU_TAG}" "GPU image"
    pull_image "${CPU_TAG}" "CPU image"
    pull_image "${DEV_TAG}" "Development image"
    pull_image "${LATEST_TAG}" "Latest image"
elif [ "$PULL_GPU" = true ]; then
    pull_image "${GPU_TAG}" "GPU image"
elif [ "$PULL_CPU" = true ]; then
    pull_image "${CPU_TAG}" "CPU image"
elif [ "$PULL_DEV" = true ]; then
    pull_image "${DEV_TAG}" "Development image"
fi

# List pulled images
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Available Images:${NC}"
echo -e "${GREEN}========================================${NC}"
docker images | head -1
docker images | grep "${DOCKER_HUB_REPO}" || echo "No images found"

echo -e "\n${GREEN}Pull complete!${NC}"
echo -e "\nNext steps:"
echo -e "  - Run training: ${YELLOW}./docker/train.sh${NC}"
echo -e "  - Start development: ${YELLOW}./docker/dev.sh${NC}"