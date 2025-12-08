#!/bin/bash
# CS627 Semantic-Vector Space - Quickstart Script
# One-command setup and demo for new users

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SAMPLE_SIZE=${SAMPLE_SIZE:-100}  # Use small sample for demo
DEMO_EPOCHS=${DEMO_EPOCHS:-3}    # Quick training for demo

echo -e "${BOLD}${BLUE}==================================================================${NC}"
echo -e "${BOLD}${BLUE}     CS627 Semantic-Vector Space - Quickstart Demo${NC}"
echo -e "${BOLD}${BLUE}==================================================================${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${BOLD}Step 1: Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found! Please install Python 3.8+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠ Docker not found. Will use local environment (may have compatibility issues)${NC}"
    USE_DOCKER=0
else
    echo -e "${GREEN}✓ Docker found${NC}"
    USE_DOCKER=1
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        echo -e "${YELLOW}⚠ Docker daemon not running. Starting Docker...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open -a Docker
            echo "Waiting for Docker to start (up to 30 seconds)..."
            for i in {1..30}; do
                if docker info &> /dev/null; then
                    echo -e "${GREEN}✓ Docker started${NC}"
                    break
                fi
                sleep 1
            done
        else
            echo -e "${RED}Please start Docker manually and run this script again${NC}"
            exit 1
        fi
    fi
fi

# Check Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}✗ Git not found! Please install git${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Git found${NC}"

echo ""

# Step 2: Setup environment
echo -e "${BOLD}Step 2: Setting up environment...${NC}"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -q --upgrade pip
pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install -q transformers datasets h5py tqdm click pillow scipy librosa

# Install CMU-MultimodalSDK if not present
if [ ! -d "CMU-MultimodalSDK" ]; then
    echo "Installing CMU-MultimodalSDK..."
    git clone -q https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
    cd CMU-MultimodalSDK && pip3 install -q . && cd ..
fi

# Create necessary directories
mkdir -p data/cmumosi/{mosi,audio,frames,videos}
mkdir -p data/pregenerated_tokens/demo
mkdir -p OptimalWeights Results/quickstart
mkdir -p configs

echo -e "${GREEN}✓ Environment setup complete${NC}"
echo ""

# Step 3: Download sample data
echo -e "${BOLD}Step 3: Preparing demo data...${NC}"

# Create a minimal demo dataset
cat > create_demo_data.py << 'EOF'
import sys
import json
import h5py
import numpy as np
from pathlib import Path

print("Creating demo dataset...")

# Create demo token files
demo_dir = Path("data/pregenerated_tokens/demo")
demo_dir.mkdir(parents=True, exist_ok=True)

# Generate random demo tokens (normally these would come from real encoders)
np.random.seed(42)

for split in ['train', 'val', 'test']:
    n_samples = {'train': 100, 'val': 20, 'test': 20}[split]
    
    with h5py.File(demo_dir / f"{split}_tokens.h5", 'w') as f:
        # Create demo tokens for each encoder
        for encoder in ['text_base', 'audio_waveform', 'audio_tone', 'image_base']:
            tokens = np.random.randn(n_samples, 1024).astype(np.float32)
            f.create_dataset(encoder, data=tokens, compression='gzip')
        
        # Add metadata
        f.create_dataset('segment_ids', 
                         data=[f"{split}_{i}".encode() for i in range(n_samples)])
        f.create_dataset('labels', 
                         data=np.random.randn(n_samples).astype(np.float32))
        
        f.attrs['num_samples'] = n_samples
        f.attrs['token_dim'] = 1024
        f.attrs['encoders'] = ['text_base', 'audio_waveform', 'audio_tone', 'image_base']
    
    print(f"  Created {split} tokens: {n_samples} samples")

# Create metadata file
metadata = {
    'train': {'num_samples': 100, 'token_dim': 1024, 
              'encoders': ['text_base', 'audio_waveform', 'audio_tone', 'image_base'],
              'file_size_mb': 1.5},
    'val': {'num_samples': 20, 'token_dim': 1024,
            'encoders': ['text_base', 'audio_waveform', 'audio_tone', 'image_base'],
            'file_size_mb': 0.3},
    'test': {'num_samples': 20, 'token_dim': 1024,
             'encoders': ['text_base', 'audio_waveform', 'audio_tone', 'image_base'],
             'file_size_mb': 0.3}
}

with open(demo_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Demo dataset created")
EOF

python3 create_demo_data.py
rm create_demo_data.py

echo -e "${GREEN}✓ Demo data prepared${NC}"
echo ""

# Step 4: Run mini training
echo -e "${BOLD}Step 4: Running mini training demo...${NC}"

if [ $USE_DOCKER -eq 1 ]; then
    echo "Building Docker image (this may take a few minutes on first run)..."
    docker build -q -f Dockerfile -t cs627-quickstart . > /dev/null 2>&1
    
    echo "Training adapter with Docker..."
    docker run --rm \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/OptimalWeights:/workspace/OptimalWeights \
        -v $(pwd)/Results:/workspace/Results \
        -e TOKEN_DIR=/workspace/data/pregenerated_tokens/demo \
        -e EPOCHS=3 \
        -e BATCH_SIZE=10 \
        cs627-quickstart \
        python src/Training/train_adapters.py --mode encoder --epochs 3 --batch-size 10
else
    echo "Training adapter locally..."
    TOKEN_DIR=data/pregenerated_tokens/demo EPOCHS=3 BATCH_SIZE=10 \
        python3 src/Training/train_adapters.py --mode encoder --epochs 3 --batch-size 10
fi

echo -e "${GREEN}✓ Training demo complete${NC}"
echo ""

# Step 5: Verify results
echo -e "${BOLD}Step 5: Verifying installation...${NC}"

# Check if weights were saved
if [ -f "OptimalWeights/text_adapter_weights.pth" ]; then
    echo -e "${GREEN}✓ Adapter weights saved successfully${NC}"
else
    echo -e "${YELLOW}⚠ No weights found (training may have failed)${NC}"
fi

# Check if results were logged
if ls Results/quickstart/*.json 1> /dev/null 2>&1 || ls Results/adapter_training/*.json 1> /dev/null 2>&1; then
    echo -e "${GREEN}✓ Training results logged${NC}"
else
    echo -e "${YELLOW}⚠ No results found${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}==================================================================${NC}"
echo -e "${BOLD}${GREEN}     Quickstart Complete! 🎉${NC}"
echo -e "${BOLD}${GREEN}==================================================================${NC}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "1. Generate real tokens from CMU-MOSI dataset:"
echo -e "   ${BLUE}make download-data    # Download dataset${NC}"
echo -e "   ${BLUE}make tokens           # Generate encoder tokens${NC}"
echo ""
echo "2. Train adapters on real data:"
echo -e "   ${BLUE}make train            # Train on GPU${NC}"
echo -e "   ${BLUE}make train-cpu        # Train on CPU${NC}"
echo ""
echo "3. Run evaluation:"
echo -e "   ${BLUE}make evaluate         # Evaluate trained models${NC}"
echo ""
echo "4. Explore with Jupyter:"
echo -e "   ${BLUE}make dev              # Start Jupyter environment${NC}"
echo ""
echo "5. Use the Python CLI for advanced workflows:"
echo -e "   ${BLUE}python cli.py --help  # Show all commands${NC}"
echo -e "   ${BLUE}python cli.py pipeline --help  # Full pipeline options${NC}"
echo ""
echo "For more information, see README.md"
echo ""

# Cleanup check
echo -e "${BOLD}Optional: Clean up demo files${NC}"
echo -e "To remove demo data and start fresh with real data, run:"
echo -e "   ${BLUE}rm -rf data/pregenerated_tokens/demo${NC}"
echo ""