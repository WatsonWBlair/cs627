#!/bin/bash

# Cloud Instance Setup Script for CS627 - Amazon Linux 2023 Version
# Modified for ec2-user and Amazon Linux 2023

# Configuration
REMOTE_USER="ec2-user"
REMOTE_HOST="$1"
REMOTE_PATH="~/cs627"
SSH_KEY="~/.ssh/SHARD_Training.pem"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if host argument is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide the remote host IP or hostname${NC}"
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 35.175.222.131"
    exit 1
fi

echo -e "${GREEN}Setting up remote instance at ${REMOTE_USER}@${REMOTE_HOST}${NC}"

# Step 1: Create directory structure on remote
echo -e "${YELLOW}Step 1: Creating directory structure on remote...${NC}"
ssh -i ${SSH_KEY} ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}/{data/cmumosi,src,scripts,OptimalWeights,tests,training_reports}"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Directory structure created${NC}"
else
    echo -e "${RED}✗ Failed to create directories${NC}"
    exit 1
fi

# Step 2: Sync source code
echo -e "${YELLOW}Step 2: Syncing source code...${NC}"
rsync -avz --progress -e "ssh -i ${SSH_KEY}" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'data/' \
    --exclude 'OptimalWeights/*.pth' \
    src/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/src/
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Source code synced${NC}"
else
    echo -e "${RED}✗ Failed to sync source code${NC}"
    exit 1
fi

# Step 3: Sync scripts
echo -e "${YELLOW}Step 3: Syncing scripts...${NC}"
rsync -avz --progress -e "ssh -i ${SSH_KEY}" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    scripts/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/scripts/
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Scripts synced${NC}"
else
    echo -e "${RED}✗ Failed to sync scripts${NC}"
    exit 1
fi

# Step 4: Sync tests
echo -e "${YELLOW}Step 4: Syncing tests...${NC}"
rsync -avz --progress -e "ssh -i ${SSH_KEY}" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    tests/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/tests/
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Tests synced${NC}"
else
    echo -e "${RED}✗ Failed to sync tests${NC}"
    exit 1
fi

# Step 5: Sync requirements and config files
echo -e "${YELLOW}Step 5: Syncing requirements and config files...${NC}"
rsync -avz --progress -e "ssh -i ${SSH_KEY}" \
    requirements.txt \
    CLAUDE.md \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Config files synced${NC}"
else
    echo -e "${RED}✗ Failed to sync config files${NC}"
    exit 1
fi

# Step 6: Sync MOSI data (if it exists locally)
if [ -d "data/cmumosi" ]; then
    echo -e "${YELLOW}Step 6: Syncing MOSI dataset (this may take a while)...${NC}"
    rsync -avz --progress -e "ssh -i ${SSH_KEY}" data/cmumosi/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/data/cmumosi/
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ MOSI data synced${NC}"
    else
        echo -e "${RED}✗ Failed to sync MOSI data${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Step 6: Skipping MOSI data sync (data/cmumosi not found locally)${NC}"
fi

# Step 7: Sync OptimalWeights (if any .pth files exist)
if ls OptimalWeights/*.pth 1> /dev/null 2>&1; then
    echo -e "${YELLOW}Step 7: Syncing model weights...${NC}"
    rsync -avz --progress -e "ssh -i ${SSH_KEY}" OptimalWeights/*.pth ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/OptimalWeights/
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model weights synced${NC}"
    else
        echo -e "${RED}✗ Failed to sync model weights${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Step 7: No model weights to sync${NC}"
fi

# Step 8: Install Python and dependencies for Amazon Linux 2023
echo -e "${YELLOW}Step 8: Setting up Python environment on Amazon Linux 2023...${NC}"
ssh -i ${SSH_KEY} ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
cd ~/cs627

# Install Python 3.11 and development tools if needed
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-pip python3.11-devel gcc gcc-c++ make

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch CPU version first (we'll check for GPU later)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project requirements
pip install transformers diffusers datasets accelerate evaluate sentence-transformers
pip install matplotlib librosa soundfile h5py python-dotenv CMU-MultimodalSDK
pip install safetensors huggingface-hub tqdm pyyaml
pip install mteb scikit-learn pytest

echo ""
echo "=== Python environment setup complete ==="
python --version
pip list | grep -E "torch|transformers|diffusers" | head -5
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python environment configured${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Some packages may not have installed correctly${NC}"
fi

# Step 9: Check for GPU and install drivers if needed
echo -e "${YELLOW}Step 9: Checking for GPU support...${NC}"
GPU_TYPE=$(ssh -i ${SSH_KEY} ${REMOTE_USER}@${REMOTE_HOST} "lspci 2>/dev/null | grep -E 'NVIDIA|AMD' | head -1")

if echo "$GPU_TYPE" | grep -q "NVIDIA"; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    echo -e "${YELLOW}Note: You may need to install NVIDIA drivers manually on Amazon Linux 2023${NC}"
    echo "To install GPU drivers, SSH into the instance and run:"
    echo "  sudo dnf install -y kernel-modules-extra"
    echo "  sudo dnf install -y nvidia-driver nvidia-driver-cuda"
    echo "  sudo reboot"
else
    echo -e "${YELLOW}No GPU detected - using CPU PyTorch${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Remote instance setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "You can now SSH into the instance and run tests:"
echo -e "${YELLOW}ssh -i ${SSH_KEY} ${REMOTE_USER}@${REMOTE_HOST}${NC}"
echo -e "${YELLOW}cd ${REMOTE_PATH}${NC}"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo -e "${YELLOW}python -m pytest tests/unit/smoke_test.py -v${NC}"