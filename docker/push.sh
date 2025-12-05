#!/bin/bash

# Docker Push Script for CS627 Semantic-Vector Space
# Pushes images to Docker Hub

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
echo -e "${GREEN}CS627 Docker Hub Push Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if user is logged in to Docker Hub
if ! docker info 2>/dev/null | grep -q "Username: ${DOCKER_HUB_USERNAME}"; then
    echo -e "${YELLOW}Not logged in to Docker Hub.${NC}"
    echo -e "${YELLOW}Logging in as ${DOCKER_HUB_USERNAME}...${NC}"
    docker login -u ${DOCKER_HUB_USERNAME}
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to login to Docker Hub${NC}"
        exit 1
    fi
fi

# Function to push an image
push_image() {
    local tag=$1
    local description=$2
    
    echo -e "\n${YELLOW}Pushing ${description}...${NC}"
    echo -e "Tag: ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${tag}"
    
    docker push ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${tag}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${description} pushed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to push ${description}${NC}"
        return 1
    fi
}

# Ask for confirmation
echo -e "\n${YELLOW}This will push the following images to Docker Hub:${NC}"
echo -e "  - ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${GPU_TAG}"
echo -e "  - ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${CPU_TAG}"
echo -e "  - ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${DEV_TAG}"
echo -e "  - ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:${LATEST_TAG}"
echo -e "\n${YELLOW}Continue? (y/N):${NC} \c"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${RED}Push cancelled${NC}"
    exit 0
fi

# Push images
push_image "${GPU_TAG}" "GPU image"
push_image "${CPU_TAG}" "CPU image"
push_image "${DEV_TAG}" "Development image"
push_image "${LATEST_TAG}" "Latest (GPU) image"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Push Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nImages are now available at:"
echo -e "  ${BLUE}https://hub.docker.com/r/${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}${NC}"
echo -e "\nOthers can pull and use with:"
echo -e "  ${YELLOW}docker pull ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:latest${NC}"
echo -e "  ${YELLOW}docker run --gpus all -v ./data:/workspace/data ${DOCKER_HUB_USERNAME}/${DOCKER_HUB_REPO}:latest${NC}"