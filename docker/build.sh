#!/bin/bash

# Docker Build Script for CS627 Semantic-Vector Space
# Builds all Docker images (GPU, CPU, Dev)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables if .env exists
if [ -f .env ]; then
    source .env
else
    # Use defaults from .env.docker
    source .env.docker
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CS627 Docker Build Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to build an image
build_image() {
    local dockerfile=$1
    local tag=$2
    local description=$3
    
    echo -e "\n${YELLOW}Building ${description}...${NC}"
    echo -e "Dockerfile: ${dockerfile}"
    echo -e "Tag: ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${tag}"
    
    docker build -f ${dockerfile} -t ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${tag} .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${description} built successfully${NC}"
    else
        echo -e "${RED}✗ Failed to build ${description}${NC}"
        exit 1
    fi
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Build GPU image
build_image "Dockerfile" "${GPU_TAG}" "GPU-enabled image"

# Tag GPU image as latest
docker tag ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${GPU_TAG} \
           ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${LATEST_TAG}
echo -e "${GREEN}✓ Tagged GPU image as latest${NC}"

# Build CPU image
build_image "Dockerfile.cpu" "${CPU_TAG}" "CPU-only image"

# Build development image
build_image "Dockerfile.dev" "${DEV_TAG}" "Development image"

# List built images
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Built Images:${NC}"
echo -e "${GREEN}========================================${NC}"
docker images | grep "${DOCKER_HUB_REPO}" | head -5

echo -e "\n${GREEN}Build complete!${NC}"
echo -e "\nNext steps:"
echo -e "  - Test locally: ${YELLOW}./docker/train.sh${NC}"
echo -e "  - Run dev environment: ${YELLOW}./docker/dev.sh${NC}"
echo -e "  - Push to Docker Hub: ${YELLOW}./docker/push.sh${NC}"