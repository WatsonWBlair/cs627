#!/bin/bash
# ==============================================================================
# Download trained model weights and results from cloud instance
# ==============================================================================
# This script downloads:
#   - OptimalWeights/ (trained adapter weights)
#   - EncoderCheckpoints/ (encoder training checkpoints)
#   - DecoderCheckpoints/ (decoder training checkpoints)
#   - results/ (training logs, evaluation metrics)
#   - training_reports/ (loss curves, metrics)
#
# Usage:
#   ./scripts/cloud/download_from_cloud.sh [OPTIONS]
#
# Options:
#   --weights-only    Download only adapter weights (default: download all)
#   --results-only    Download only results and logs
#   --dry-run         Show what would be transferred without actually transferring
#   --help            Show this help message
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ==============================================================================
# Load environment variables from .env
# ==============================================================================

ENV_FILE="$PROJECT_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo -e "${YELLOW}Please copy .env.example to .env and configure your cloud instance details:${NC}"
    echo -e "  cp .env.example .env"
    echo -e "  # Then edit .env with your cloud instance info"
    exit 1
fi

# Load .env file
export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)

# Validate required variables
if [ -z "$CLOUD_USER" ] || [ -z "$CLOUD_HOST" ]; then
    echo -e "${RED}Error: CLOUD_USER and CLOUD_HOST must be set in .env${NC}"
    exit 1
fi

# Default values
CLOUD_PROJECT_DIR="${CLOUD_PROJECT_DIR:-~/cs627}"
OPTIMAL_WEIGHTS_DIR="${OPTIMAL_WEIGHTS_DIR:-OptimalWeights}"
ENCODER_CHECKPOINT_DIR="${ENCODER_CHECKPOINT_DIR:-EncoderCheckpoints}"
DECODER_CHECKPOINT_DIR="${DECODER_CHECKPOINT_DIR:-DecoderCheckpoints}"
RESULTS_DIR="${RESULTS_DIR:-results}"
TRAINING_REPORTS_DIR="${TRAINING_REPORTS_DIR:-training_reports}"
CLOUD_SSH_KEY="${CLOUD_SSH_KEY}"
CLOUD_PORT="${CLOUD_PORT:-22}"

# Parse command line arguments
DOWNLOAD_WEIGHTS=true
DOWNLOAD_RESULTS=true
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --weights-only)
            DOWNLOAD_RESULTS=false
            shift
            ;;
        --results-only)
            DOWNLOAD_WEIGHTS=false
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --help)
            head -n 21 "$0" | tail -n 18
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Build SSH/rsync options
# ==============================================================================

SSH_OPTS="-p $CLOUD_PORT"
RSYNC_OPTS="-avz --progress"

if [ -n "$CLOUD_SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $CLOUD_SSH_KEY"
    RSYNC_OPTS="$RSYNC_OPTS -e 'ssh -p $CLOUD_PORT -i $CLOUD_SSH_KEY'"
else
    RSYNC_OPTS="$RSYNC_OPTS -e 'ssh -p $CLOUD_PORT'"
fi

if [ -n "$DRY_RUN" ]; then
    RSYNC_OPTS="$RSYNC_OPTS $DRY_RUN"
fi

CLOUD_CONN="${CLOUD_USER}@${CLOUD_HOST}"

# ==============================================================================
# Create instance-specific download directory
# ==============================================================================

# Generate unique directory name with timestamp and hostname
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME_SHORT=$(echo "$CLOUD_HOST" | cut -d'.' -f1)
DOWNLOAD_DIR="$PROJECT_ROOT/CandidateWeights/${HOSTNAME_SHORT}_${TIMESTAMP}"

# Create directory structure
mkdir -p "$DOWNLOAD_DIR/OptimalWeights"
mkdir -p "$DOWNLOAD_DIR/EncoderCheckpoints"
mkdir -p "$DOWNLOAD_DIR/DecoderCheckpoints"
mkdir -p "$DOWNLOAD_DIR/results"
mkdir -p "$DOWNLOAD_DIR/training_reports"

# ==============================================================================
# Print configuration
# ==============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  Download from Cloud Instance                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Cloud Instance:${NC} $CLOUD_CONN"
echo -e "${YELLOW}Remote Directory:${NC} $CLOUD_PROJECT_DIR"
echo -e "${YELLOW}Download To:${NC} $DOWNLOAD_DIR"
echo ""

if [ -n "$DRY_RUN" ]; then
    echo -e "${YELLOW}[WARN] DRY RUN MODE - No files will be transferred${NC}"
    echo ""
fi

# ==============================================================================
# Test SSH connection
# ==============================================================================

echo -e "${BLUE}[1/6] Testing SSH connection...${NC}"
if ssh $SSH_OPTS -o ConnectTimeout=10 "$CLOUD_CONN" "echo 'Connection successful'" > /dev/null 2>&1; then
    echo -e "${GREEN}[OK] SSH connection successful${NC}"
else
    echo -e "${RED}[ERR] Failed to connect to $CLOUD_CONN${NC}"
    echo -e "${YELLOW}Check your .env configuration and network connectivity${NC}"
    exit 1
fi
echo ""

# ==============================================================================
# Download adapter weights (ALWAYS)
# ==============================================================================

echo -e "${BLUE}[2/6] Downloading trained adapter weights...${NC}"

# Check if OptimalWeights directory exists on remote
if ssh $SSH_OPTS "$CLOUD_CONN" "[ -d '$CLOUD_PROJECT_DIR/$OPTIMAL_WEIGHTS_DIR' ]"; then
    eval rsync $RSYNC_OPTS \
        "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$OPTIMAL_WEIGHTS_DIR/" \
        "$DOWNLOAD_DIR/OptimalWeights/"

    if [ -z "$DRY_RUN" ]; then
        # Count downloaded weights
        WEIGHT_COUNT=$(find "$DOWNLOAD_DIR/OptimalWeights" -name "*.pth" 2>/dev/null | wc -l)
        echo -e "${GREEN}[OK] Downloaded $WEIGHT_COUNT adapter weight files${NC}"
    fi
else
    echo -e "${YELLOW}[WARN] No OptimalWeights directory found on remote instance${NC}"
    echo -e "${YELLOW}   Training may not have completed or saved weights${NC}"
fi
echo ""

# ==============================================================================
# Download checkpoints
# ==============================================================================

echo -e "${BLUE}[3/6] Downloading encoder checkpoints...${NC}"

if ssh $SSH_OPTS "$CLOUD_CONN" "[ -d '$CLOUD_PROJECT_DIR/$ENCODER_CHECKPOINT_DIR' ]"; then
    eval rsync $RSYNC_OPTS \
        "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$ENCODER_CHECKPOINT_DIR/" \
        "$DOWNLOAD_DIR/EncoderCheckpoints/"

    if [ -z "$DRY_RUN" ]; then
        CKPT_COUNT=$(find "$DOWNLOAD_DIR/EncoderCheckpoints" -name "*.json" 2>/dev/null | wc -l)
        echo -e "${GREEN}[OK] Downloaded $CKPT_COUNT encoder checkpoint files${NC}"
    fi
else
    echo -e "${YELLOW}[WARN] No EncoderCheckpoints directory found${NC}"
fi
echo ""

echo -e "${BLUE}[4/6] Downloading decoder checkpoints...${NC}"

if ssh $SSH_OPTS "$CLOUD_CONN" "[ -d '$CLOUD_PROJECT_DIR/$DECODER_CHECKPOINT_DIR' ]"; then
    eval rsync $RSYNC_OPTS \
        "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$DECODER_CHECKPOINT_DIR/" \
        "$DOWNLOAD_DIR/DecoderCheckpoints/"

    if [ -z "$DRY_RUN" ]; then
        CKPT_COUNT=$(find "$DOWNLOAD_DIR/DecoderCheckpoints" -name "*.json" 2>/dev/null | wc -l)
        echo -e "${GREEN}[OK] Downloaded $CKPT_COUNT decoder checkpoint files${NC}"
    fi
else
    echo -e "${YELLOW}[WARN] No DecoderCheckpoints directory found${NC}"
fi
echo ""

# ==============================================================================
# Download results and logs
# ==============================================================================

if [ "$DOWNLOAD_RESULTS" = true ]; then
    echo -e "${BLUE}[5/6] Downloading training results...${NC}"

    # Download results directory
    if ssh $SSH_OPTS "$CLOUD_CONN" "[ -d '$CLOUD_PROJECT_DIR/$RESULTS_DIR' ]"; then
        eval rsync $RSYNC_OPTS \
            "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$RESULTS_DIR/" \
            "$DOWNLOAD_DIR/results/"

        if [ -z "$DRY_RUN" ]; then
            echo -e "${GREEN}[OK] Results downloaded${NC}"
        fi
    else
        echo -e "${YELLOW}[WARN] No results directory found${NC}"
    fi
    echo ""

    echo -e "${BLUE}[6/6] Downloading training reports...${NC}"

    # Download training_reports directory (if it exists)
    if ssh $SSH_OPTS "$CLOUD_CONN" "[ -d '$CLOUD_PROJECT_DIR/$TRAINING_REPORTS_DIR' ]"; then
        eval rsync $RSYNC_OPTS \
            "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$TRAINING_REPORTS_DIR/" \
            "$DOWNLOAD_DIR/training_reports/"

        if [ -z "$DRY_RUN" ]; then
            echo -e "${GREEN}[OK] Training reports downloaded${NC}"
        fi
    else
        echo -e "${YELLOW}[WARN] No training_reports directory found${NC}"
    fi
    echo ""
else
    echo -e "${BLUE}[5/6] Skipping results (--weights-only specified)${NC}"
    echo -e "${BLUE}[6/6] Skipping training reports (--weights-only specified)${NC}"
    echo ""
fi

# ==============================================================================
# Summary
# ==============================================================================

if [ -z "$DRY_RUN" ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                      Download Complete!                                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Downloaded to instance-specific directory:${NC}"
    echo -e "  ${GREEN}$DOWNLOAD_DIR/${NC}"
    echo ""
    echo -e "${YELLOW}Contents:${NC}"
    echo -e "  [+] Adapter weights: ${GREEN}$DOWNLOAD_DIR/OptimalWeights/${NC}"
    echo -e "  [+] Encoder checkpoints: ${GREEN}$DOWNLOAD_DIR/EncoderCheckpoints/${NC}"
    echo -e "  [+] Decoder checkpoints: ${GREEN}$DOWNLOAD_DIR/DecoderCheckpoints/${NC}"

    if [ "$DOWNLOAD_RESULTS" = true ]; then
        echo -e "  [+] Results: ${GREEN}$DOWNLOAD_DIR/results/${NC}"
        echo -e "  [+] Training reports: ${GREEN}$DOWNLOAD_DIR/training_reports/${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Quick Stats:${NC}"
    if [ -d "$DOWNLOAD_DIR/OptimalWeights" ]; then
        WEIGHT_FILES=$(find "$DOWNLOAD_DIR/OptimalWeights" -name "*.pth" 2>/dev/null | wc -l)
        echo -e "  Adapter weights: ${GREEN}$WEIGHT_FILES files${NC}"
    fi
    if [ -f "$DOWNLOAD_DIR/training_reports/training_metrics.json" ]; then
        echo -e "  Training metrics: ${GREEN}Available${NC}"
    fi
    if [ -f "$DOWNLOAD_DIR/training_reports/loss_curves.png" ]; then
        echo -e "  Loss curves: ${GREEN}Available${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo ""
    echo -e "${BLUE}1. Review training metrics:${NC}"
    echo -e "   cat $DOWNLOAD_DIR/training_reports/training_metrics.json"
    echo ""
    echo -e "${BLUE}2. View loss curves:${NC}"
    echo -e "   open $DOWNLOAD_DIR/training_reports/loss_curves.png"
    echo ""
    echo -e "${BLUE}3. Promote best weights to production (if satisfied):${NC}"
    echo -e "   cp $DOWNLOAD_DIR/OptimalWeights/* OptimalWeights/"
    echo ""
    echo -e "${BLUE}4. Or compare with existing weights:${NC}"
    echo -e "   python scripts/cloud/compare_weights.py $DOWNLOAD_DIR/OptimalWeights/ OptimalWeights/"
    echo ""
    echo -e "${BLUE}5. Run evaluation on cloud-trained weights:${NC}"
    echo -e "   python scripts/run_evaluation.py --weights-dir $DOWNLOAD_DIR/OptimalWeights/"
    echo ""
    echo -e "${YELLOW}All cloud-trained weights are saved separately and won't overwrite your local weights.${NC}"
else
    echo -e "${YELLOW}Dry run complete. No files were transferred.${NC}"
    echo -e "${YELLOW}Run without --dry-run to actually download files.${NC}"
fi
