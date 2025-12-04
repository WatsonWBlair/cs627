#!/bin/bash

################################################################################
# Cloud Instance Setup and Training Script
################################################################################
#
# This script automates the complete setup and training process for the
# Semantic-Vector Space multimodal encoder alignment project.
#
# Intended for use on cloud GPU instances (AWS, GCP, Azure, Lambda Labs, etc.)
#
# Usage:
#   chmod +x setup_and_train.sh
#   ./setup_and_train.sh
#
# Or with custom parameters:
#   ./setup_and_train.sh --batch-size 64 --queue-size 16384 --epochs 200
#
################################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
BATCH_SIZE=32
QUEUE_SIZE=4096
NUM_EPOCHS=100
SKIP_SETUP=false
SKIP_DATA=false
SKIP_TRAIN=false
DATA_PATH="data/cmumosi"
OUTPUT_DIR="results/encoder_alignment"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --queue-size)
            QUEUE_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --batch-size SIZE      Batch size (default: 32)"
            echo "  --queue-size SIZE      Memory queue size (default: 4096)"
            echo "  --epochs NUM           Number of training epochs (default: 100)"
            echo "  --skip-setup           Skip dependency installation"
            echo "  --skip-data            Skip data download/preprocessing"
            echo "  --skip-train           Skip training (setup only)"
            echo "  --data-path PATH       Data directory (default: data/cmumosi)"
            echo "  --output-dir DIR       Output directory (default: results/encoder_alignment)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Print Configuration
################################################################################

echo "================================================================================"
echo "  Cloud Instance Setup and Training"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Batch Size:       $BATCH_SIZE"
echo "  Queue Size:       $QUEUE_SIZE"
echo "  Epochs:           $NUM_EPOCHS"
echo "  Data Path:        $DATA_PATH"
echo "  Output Dir:       $OUTPUT_DIR"
echo "  Skip Setup:       $SKIP_SETUP"
echo "  Skip Data:        $SKIP_DATA"
echo "  Skip Training:    $SKIP_TRAIN"
echo ""
echo "================================================================================"
echo ""

################################################################################
# System Information
################################################################################

log_info "Gathering system information..."
echo ""
echo "Hostname:     $(hostname)"
echo "OS:           $(uname -s)"
echo "Kernel:       $(uname -r)"
echo "Architecture: $(uname -m)"
echo "CPU Cores:    $(nproc)"
echo "Memory:       $(free -h | grep Mem | awk '{print $2}')"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    log_success "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo ""
    log_warning "No GPU detected (nvidia-smi not found)"
fi

echo ""

################################################################################
# Step 1: Install System Dependencies
################################################################################

if [ "$SKIP_SETUP" = false ]; then
    log_info "Step 1: Installing system dependencies..."

    # Detect package manager
    if command -v apt-get &> /dev/null; then
        log_info "Detected apt package manager (Debian/Ubuntu)"
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            python3 \
            python3-pip \
            python3-venv \
            git \
            wget \
            curl \
            build-essential \
            libsndfile1 \
            ffmpeg
    elif command -v yum &> /dev/null; then
        log_info "Detected yum package manager (RHEL/CentOS)"
        sudo yum install -y -q \
            python3 \
            python3-pip \
            git \
            wget \
            curl \
            gcc \
            gcc-c++ \
            make \
            libsndfile \
            ffmpeg
    else
        log_warning "Unknown package manager - skipping system dependency installation"
        log_warning "Please ensure python3, pip, git, libsndfile, and ffmpeg are installed"
    fi

    log_success "System dependencies installed"
    echo ""
else
    log_info "Skipping system dependency installation (--skip-setup)"
    echo ""
fi

################################################################################
# Step 2: Setup Python Environment
################################################################################

if [ "$SKIP_SETUP" = false ]; then
    log_info "Step 2: Setting up Python environment..."

    # Upgrade pip
    log_info "Upgrading pip..."
    python3 -m pip install --upgrade pip --quiet

    # Install Python dependencies
    log_info "Installing Python dependencies from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet
        log_success "Python dependencies installed"
    else
        log_error "requirements.txt not found!"
        exit 1
    fi

    echo ""
else
    log_info "Skipping Python environment setup (--skip-setup)"
    echo ""
fi

################################################################################
# Step 3: Install CMU-MultimodalSDK
################################################################################

if [ "$SKIP_SETUP" = false ]; then
    log_info "Step 3: Installing CMU-MultimodalSDK..."

    if [ ! -d "CMU-MultimodalSDK" ]; then
        log_info "Cloning CMU-MultimodalSDK repository..."
        git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git --quiet
    else
        log_info "CMU-MultimodalSDK directory already exists, skipping clone"
    fi

    log_info "Installing CMU-MultimodalSDK..."
    cd CMU-MultimodalSDK
    pip install . --quiet
    cd ..

    log_success "CMU-MultimodalSDK installed"
    echo ""
else
    log_info "Skipping CMU-MultimodalSDK installation (--skip-setup)"
    echo ""
fi

################################################################################
# Step 4: Verify Installation
################################################################################

if [ "$SKIP_SETUP" = false ]; then
    log_info "Step 4: Verifying installation..."

    # Check Python packages
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || { log_error "PyTorch not installed!"; exit 1; }
    python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || { log_error "Transformers not installed!"; exit 1; }

    # Check CUDA availability
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        log_success "CUDA available (version: $CUDA_VERSION)"
        NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
        log_info "Number of GPUs: $NUM_GPUS"
    else
        log_warning "CUDA not available - training will use CPU (slow!)"
    fi

    echo ""
else
    log_info "Skipping installation verification (--skip-setup)"
    echo ""
fi

################################################################################
# Step 5: Download and Preprocess Data
################################################################################

if [ "$SKIP_DATA" = false ]; then
    log_info "Step 5: Downloading and preprocessing CMU-MOSI dataset..."

    # Create data directory
    mkdir -p "$DATA_PATH"

    # Check if preprocessed data already exists
    if [ -f "$DATA_PATH/preprocessed_train.pkl" ]; then
        log_info "Preprocessed data found at $DATA_PATH"
        read -p "Do you want to re-download and preprocess? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping data download/preprocessing"
        else
            log_info "Downloading CMU-MOSI metadata (this may take a while)..."
            python3 -c "
from src.Training.Data_Wrangling.mosi_dataset import download_mosi
download_mosi('$DATA_PATH/mosi/')
"
            log_success "Dataset metadata downloaded"
        fi
    else
        log_info "Downloading CMU-MOSI metadata (this may take a while)..."
        python3 -c "
from src.Training.Data_Wrangling.mosi_dataset import download_mosi
download_mosi('$DATA_PATH/mosi/')
"
        log_success "Dataset metadata downloaded"
    fi

    echo ""
else
    log_info "Skipping data download/preprocessing (--skip-data)"
    echo ""
fi

################################################################################
# Step 6: Run Training
################################################################################

if [ "$SKIP_TRAIN" = false ]; then
    log_info "Step 6: Starting training..."
    echo ""
    log_info "Training configuration:"
    log_info "  Batch size:  $BATCH_SIZE"
    log_info "  Queue size:  $QUEUE_SIZE"
    log_info "  Epochs:      $NUM_EPOCHS"
    log_info "  Output dir:  $OUTPUT_DIR"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Update training script configuration (create a temporary config file)
    cat > /tmp/training_config.py <<EOF
# Auto-generated training configuration
BATCH_SIZE = $BATCH_SIZE
QUEUE_SIZE = $QUEUE_SIZE
NUM_EPOCHS = $NUM_EPOCHS
DATA_PATH = '$DATA_PATH'
OUTPUT_DIR = '$OUTPUT_DIR'
EOF

    # Run training with logging
    LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
    log_info "Training logs will be saved to: $LOG_FILE"
    echo ""

    # Start training (with tee to show output and save to log)
    python3 src/Training/train_encoders.py 2>&1 | tee "$LOG_FILE"

    TRAIN_EXIT_CODE=${PIPESTATUS[0]}

    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "Training completed successfully!"
    else
        log_error "Training failed with exit code: $TRAIN_EXIT_CODE"
        log_error "Check logs at: $LOG_FILE"
        exit $TRAIN_EXIT_CODE
    fi

    echo ""
else
    log_info "Skipping training (--skip-train)"
    echo ""
fi

################################################################################
# Step 7: Summary
################################################################################

log_info "Step 7: Summary"
echo ""
echo "================================================================================"
echo "  Setup and Training Complete!"
echo "================================================================================"
echo ""
echo "Trained adapter weights saved to:"
echo "  AdapterWeights/"
echo ""
echo "Training results saved to:"
echo "  $OUTPUT_DIR"
echo ""
echo "Training logs:"
if [ "$SKIP_TRAIN" = false ]; then
    echo "  $LOG_FILE"
else
    echo "  (training was skipped)"
fi
echo ""

# Show adapter weight files
if [ -d "AdapterWeights" ]; then
    log_info "Adapter weights:"
    ls -lh AdapterWeights/*.pth 2>/dev/null || log_warning "No adapter weights found"
    echo ""
fi

# Show GPU memory usage (if available)
if command -v nvidia-smi &> /dev/null; then
    log_info "Final GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    echo ""
fi

log_success "All done! Your encoders are ready for use."
echo ""
echo "To use the trained encoders:"
echo "  python3 -c \"from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec; encoder = Text_to_Vec()\""
echo ""
echo "================================================================================"
