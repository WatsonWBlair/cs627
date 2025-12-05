#!/bin/bash

# Cloud Instance Setup Script for CS627 Semantic-Vector Space Training
# 
# RECOMMENDED INSTANCE CONFIGURATION:
# - Instance Type: g5.4xlarge (NVIDIA A10G GPU, 24GB VRAM)
# - AMI: AWS Deep Learning OSS Nvidia Driver AMI (PyTorch 2.5)
#   - AMI ID (us-east-1): ami-04f3e35dc85e9423b
#   - AMI ID (us-west-2): ami-0c6f4e57e7d5e5e3b
# - Storage: 200GB gp3 SSD
# 
# LAUNCH COMMAND:
# aws ec2 run-instances \
#   --image-id ami-04f3e35dc85e9423b \
#   --instance-type g5.4xlarge \
#   --key-name your-key-name \
#   --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}"
#
# USAGE:
# ./scripts/setup_remote_instance.sh <instance-ip>
#
# NOTE: This script is optimized for Deep Learning AMIs. For standard Ubuntu AMIs,
#       additional driver installation and reboots may be required.

# Configuration
REMOTE_HOST="$1"
REMOTE_PATH="~/cs627"

# Auto-detect the correct user for the instance
# Try ubuntu first (for Ubuntu AMIs), then ec2-user (for Amazon Linux)
echo "Detecting remote user..."
if ssh -i ~/.ssh/SHARD_Training.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@${REMOTE_HOST} "echo 'ubuntu'" 2>/dev/null; then
    REMOTE_USER="ubuntu"
    echo "Using user: ubuntu"
elif ssh -i ~/.ssh/SHARD_Training.pem -o ConnectTimeout=5 -o StrictHostKeyChecking=no ec2-user@${REMOTE_HOST} "echo 'ec2-user'" 2>/dev/null; then
    REMOTE_USER="ec2-user"
    echo "Using user: ec2-user"
else
    echo -e "${RED}Error: Could not connect with ubuntu or ec2-user${NC}"
    echo "Please check your instance and SSH key"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if host argument is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide the remote host IP or hostname${NC}"
    echo "Usage: $0 <remote_host>"
    echo "Example: $0 18.212.238.86"
    exit 1
fi

echo -e "${GREEN}Setting up remote instance at ${REMOTE_USER}@${REMOTE_HOST}${NC}"

# Step 1: Create directory structure on remote
echo -e "${YELLOW}Step 1: Creating directory structure on remote...${NC}"
ssh -i ~/.ssh/SHARD_Training.pem ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}/{data/cmumosi,src,scripts,OptimalWeights,tests}"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Directory structure created${NC}"
else
    echo -e "${RED}✗ Failed to create directories${NC}"
    exit 1
fi

# Step 2: Sync source code
echo -e "${YELLOW}Step 2: Syncing source code...${NC}"
rsync -avz --progress -e "ssh -i ~/.ssh/SHARD_Training.pem" \
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
rsync -avz --progress -e "ssh -i ~/.ssh/SHARD_Training.pem" \
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
rsync -avz --progress -e "ssh -i ~/.ssh/SHARD_Training.pem" \
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
rsync -avz --progress -e "ssh -i ~/.ssh/SHARD_Training.pem" \
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
    rsync -avz --progress -e "ssh -i ~/.ssh/SHARD_Training.pem" data/cmumosi/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/data/cmumosi/
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
    rsync -avz --progress -e "ssh -i ~/.ssh/SHARD_Training.pem" OptimalWeights/*.pth ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/OptimalWeights/
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model weights synced${NC}"
    else
        echo -e "${RED}✗ Failed to sync model weights${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Step 7: No model weights to sync${NC}"
fi

# Step 8: Clean up disk space and prepare Python environment
echo -e "${YELLOW}Step 8: Cleaning up disk space...${NC}"
ssh -i ~/.ssh/SHARD_Training.pem ${REMOTE_USER}@${REMOTE_HOST} "sudo apt clean 2>/dev/null || true"
echo -e "${GREEN}✓ Disk space cleaned${NC}"

# Step 9: Install Python venv package if needed
echo -e "${YELLOW}Step 9: Checking Python virtual environment support...${NC}"
ssh -i ~/.ssh/SHARD_Training.pem ${REMOTE_USER}@${REMOTE_HOST} "python3 -m venv --help > /dev/null 2>&1 || sudo apt install -y python3-venv 2>/dev/null || sudo yum install -y python3-virtualenv 2>/dev/null"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python venv support available${NC}"
else
    echo -e "${RED}✗ Failed to setup Python venv${NC}"
    exit 1
fi

# Step 10: Detect and install GPU drivers if needed
echo -e "${YELLOW}Step 10: Checking for GPU support...${NC}"
GPU_TYPE=$(ssh -i ~/.ssh/SHARD_Training.pem ${REMOTE_USER}@${REMOTE_HOST} "lspci 2>/dev/null | grep -E 'NVIDIA|AMD' | head -1")

if echo "$GPU_TYPE" | grep -q "NVIDIA"; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    echo -e "${YELLOW}Installing NVIDIA drivers and CUDA toolkit...${NC}"
    ssh -i ~/.ssh/SHARD_Training.pem ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
# Check if nvidia-smi already works
if ! nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt update -qq
    # Install NVIDIA driver 580 for A10G compatibility
    sudo apt install -y nvidia-driver-580 nvidia-utils-580 nvidia-cuda-toolkit
    
    # Check if driver loaded immediately
    if ! nvidia-smi &> /dev/null; then
        echo "NVIDIA drivers installed but require reboot to load."
        echo "After this script completes, please run: sudo reboot"
        echo "Then re-run this script to complete Python setup."
        NEEDS_REBOOT=1
    fi
else
    echo "NVIDIA drivers already installed and loaded"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi
EOF
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"  # CUDA 12.1
elif echo "$GPU_TYPE" | grep -q "AMD"; then
    echo -e "${YELLOW}⚠ AMD GPU detected - limited ML support${NC}"
    echo -e "${YELLOW}AMD GPUs require ROCm which has limited support on Ubuntu 24.04${NC}"
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"  # Use CPU version for AMD
else
    echo -e "${YELLOW}No GPU detected - using CPU PyTorch${NC}"
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# Step 11: Check if reboot is needed for GPU drivers
if ssh -i ~/.ssh/SHARD_Training.pem ${REMOTE_USER}@${REMOTE_HOST} "test -f /tmp/needs_gpu_reboot"; then
    echo -e "${RED}GPU drivers require reboot. Please run:${NC}"
    echo -e "${YELLOW}ssh ${REMOTE_USER}@${REMOTE_HOST} 'sudo reboot'${NC}"
    echo -e "Then re-run this script after the instance restarts."
    exit 1
fi

# Step 12: Create virtual environment and install dependencies
echo -e "${YELLOW}Step 12: Setting up Python environment...${NC}"

# Check if Deep Learning AMI (has conda)
IS_DL_AMI=$(ssh -i ~/.ssh/SHARD_Training.pem ${REMOTE_USER}@${REMOTE_HOST} "which conda 2>/dev/null && echo 'true' || echo 'false'")

if [ "$IS_DL_AMI" = "true" ]; then
    echo -e "${GREEN}✓ Deep Learning AMI detected - using pre-configured environment${NC}"
    ssh -i ~/.ssh/SHARD_Training.pem ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
cd ~/cs627

# Deep Learning AMI: Use existing PyTorch environment
source activate pytorch

# Just install additional project requirements
pip install transformers diffusers datasets accelerate evaluate sentence-transformers
pip install matplotlib librosa soundfile h5py python-dotenv CMU-MultimodalSDK
EOF
else
    echo -e "${YELLOW}Standard AMI detected - creating virtual environment${NC}"
    ssh ${REMOTE_USER}@${REMOTE_HOST} << EOF
cd ~/cs627

# Standard AMI: Create and use venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install PyTorch with appropriate backend
echo "Installing PyTorch from ${PYTORCH_INDEX}"
pip install --quiet --upgrade pip
pip install torch torchvision torchaudio --index-url ${PYTORCH_INDEX}

# Install core ML packages
echo "Installing ML packages..."
pip install transformers diffusers datasets accelerate evaluate sentence-transformers

# Install additional requirements
pip install safetensors huggingface-hub tqdm pyyaml

echo ""
echo "=== Installed ML packages ==="
pip list | grep -E "torch|transformers|diffusers|accelerate" | head -5

# Test GPU availability
echo ""
echo "=== Testing GPU support ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick GPU test
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print('GPU computation test: PASSED')
else:
    print('Running in CPU mode')
"
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python environment configured${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Some packages may not have installed correctly${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Remote instance setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "You can now SSH into the instance and start training:"
echo -e "${YELLOW}ssh ${REMOTE_USER}@${REMOTE_HOST}${NC}"
echo -e "${YELLOW}cd ${REMOTE_PATH}${NC}"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo -e "${YELLOW}python src/Training/train_encoders.py${NC}"
echo ""
if echo "$GPU_TYPE" | grep -q "NVIDIA"; then
    echo "Note: NVIDIA GPU detected and configured with CUDA PyTorch."
    echo "If nvidia-smi doesn't work, reboot the instance: sudo reboot"
elif echo "$GPU_TYPE" | grep -q "AMD"; then
    echo "Note: AMD GPU detected but using CPU PyTorch (limited AMD GPU support)."
else
    echo "Note: No GPU detected - using CPU-only PyTorch."
fi