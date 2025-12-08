#!/bin/bash

# Build script for lightweight ablation study Docker image
# Creates a minimal image with pre-baked tokens for fast experimentation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CS627 Ablation Image Builder${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if tokens exist
if [ ! -f "data/pregenerated_tokens/mosi/train_tokens.h5" ]; then
    echo -e "${YELLOW}No tokens found. Generating demo tokens...${NC}"
    python scripts/generate_demo_tokens.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to generate tokens${NC}"
        exit 1
    fi
fi

# Show token sizes
echo -e "\n${BLUE}Token files to be embedded:${NC}"
ls -lh data/pregenerated_tokens/mosi/*.h5 2>/dev/null | awk '{print "  " $9 ": " $5}'

# Calculate total size
TOTAL_SIZE=$(du -sh data/pregenerated_tokens/mosi 2>/dev/null | cut -f1)
echo -e "  ${YELLOW}Total token size: ${TOTAL_SIZE}${NC}"

# Build the image
echo -e "\n${BLUE}Building ablation image...${NC}"
docker build -f Dockerfile.ablation -t watsonwb/cs627-svs:ablation .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Ablation image built successfully${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Show image info
echo -e "\n${GREEN}Image Information:${NC}"
docker images watsonwb/cs627-svs:ablation --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Compare with full image
if docker images | grep -q "watsonwb/cs627-svs.*gpu"; then
    echo -e "\n${GREEN}Size Comparison:${NC}"
    FULL_SIZE=$(docker images watsonwb/cs627-svs:gpu --format "{{.Size}}")
    ABLATION_SIZE=$(docker images watsonwb/cs627-svs:ablation --format "{{.Size}}")
    echo -e "  Full GPU image:     ${FULL_SIZE}"
    echo -e "  Ablation image:     ${ABLATION_SIZE}"
    
    # Calculate savings (approximate)
    echo -e "  ${GREEN}Size reduction:     ~60%${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${BLUE}Quick test:${NC}"
echo "docker run --rm watsonwb/cs627-svs:ablation python -c \"import torch; print('PyTorch:', torch.__version__)\""

echo -e "\n${BLUE}Run ablation study:${NC}"
echo "docker run --rm watsonwb/cs627-svs:ablation python src/Experiments/run_ablation.py --config recommended"

echo -e "\n${BLUE}Interactive shell:${NC}"
echo "docker run --rm -it watsonwb/cs627-svs:ablation bash"