#!/bin/bash
# ==============================================================================
# Download trained model weights and results from cloud instance
# ==============================================================================
# This script downloads:
#   - AdapterWeights/ (trained adapter weights)
#   - results/ (training logs, evaluation metrics)
#   - training_reports/ (loss curves, checkpoints)
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
ADAPTER_WEIGHTS_DIR="${ADAPTER_WEIGHTS_DIR:-AdapterWeights}"
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
# Print configuration
# ==============================================================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  Download from Cloud Instance                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Cloud Instance:${NC} $CLOUD_CONN"
echo -e "${YELLOW}Remote Directory:${NC} $CLOUD_PROJECT_DIR"
echo -e "${YELLOW}Local Directory:${NC} $PROJECT_ROOT"
echo ""

if [ -n "$DRY_RUN" ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - No files will be transferred${NC}"
    echo ""
fi

# ==============================================================================
# Test SSH connection
# ==============================================================================

echo -e "${BLUE}[1/3] Testing SSH connection...${NC}"
if ssh $SSH_OPTS -o ConnectTimeout=10 "$CLOUD_CONN" "echo 'Connection successful'" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ SSH connection successful${NC}"
else
    echo -e "${RED}✗ Failed to connect to $CLOUD_CONN${NC}"
    echo -e "${YELLOW}Check your .env configuration and network connectivity${NC}"
    exit 1
fi
echo ""

# ==============================================================================
# Download adapter weights
# ==============================================================================

if [ "$DOWNLOAD_WEIGHTS" = true ]; then
    echo -e "${BLUE}[2/3] Downloading adapter weights...${NC}"

    # Check if AdapterWeights directory exists on remote
    if ssh $SSH_OPTS "$CLOUD_CONN" "[ -d '$CLOUD_PROJECT_DIR/$ADAPTER_WEIGHTS_DIR' ]"; then
        mkdir -p "$PROJECT_ROOT/$ADAPTER_WEIGHTS_DIR"

        eval rsync $RSYNC_OPTS \
            "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$ADAPTER_WEIGHTS_DIR/" \
            "$PROJECT_ROOT/$ADAPTER_WEIGHTS_DIR/"

        if [ -z "$DRY_RUN" ]; then
            echo -e "${GREEN}✓ Adapter weights downloaded${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No AdapterWeights directory found on remote instance${NC}"
    fi
    echo ""
else
    echo -e "${BLUE}[2/3] Skipping adapter weights (--results-only specified)${NC}"
    echo ""
fi

# ==============================================================================
# Download results and logs
# ==============================================================================

if [ "$DOWNLOAD_RESULTS" = true ]; then
    echo -e "${BLUE}[3/3] Downloading training results and logs...${NC}"

    # Download results directory
    if ssh $SSH_OPTS "$CLOUD_CONN" "[ -d '$CLOUD_PROJECT_DIR/$RESULTS_DIR' ]"; then
        mkdir -p "$PROJECT_ROOT/$RESULTS_DIR"

        eval rsync $RSYNC_OPTS \
            "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$RESULTS_DIR/" \
            "$PROJECT_ROOT/$RESULTS_DIR/"

        if [ -z "$DRY_RUN" ]; then
            echo -e "${GREEN}✓ Results downloaded${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No results directory found on remote instance${NC}"
    fi

    # Download training_reports directory (if it exists)
    if ssh $SSH_OPTS "$CLOUD_CONN" "[ -d '$CLOUD_PROJECT_DIR/$TRAINING_REPORTS_DIR' ]"; then
        mkdir -p "$PROJECT_ROOT/$TRAINING_REPORTS_DIR"

        eval rsync $RSYNC_OPTS \
            "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$TRAINING_REPORTS_DIR/" \
            "$PROJECT_ROOT/$TRAINING_REPORTS_DIR/"

        if [ -z "$DRY_RUN" ]; then
            echo -e "${GREEN}✓ Training reports downloaded${NC}"
        fi
    fi
    echo ""
else
    echo -e "${BLUE}[3/3] Skipping results (--weights-only specified)${NC}"
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
    echo -e "${YELLOW}Downloaded artifacts:${NC}"

    if [ "$DOWNLOAD_WEIGHTS" = true ]; then
        echo -e "  📦 Adapter weights: ${GREEN}$ADAPTER_WEIGHTS_DIR/${NC}"
    fi

    if [ "$DOWNLOAD_RESULTS" = true ]; then
        echo -e "  📊 Results: ${GREEN}$RESULTS_DIR/${NC}"
        echo -e "  📈 Training reports: ${GREEN}$TRAINING_REPORTS_DIR/${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Use trained encoders with: ${BLUE}from Encoders import Text_to_Vec, Audio_to_Vec${NC}"
    echo -e "  2. Review training logs in: ${BLUE}$RESULTS_DIR/${NC}"
    echo -e "  3. Run evaluation: ${BLUE}python scripts/run_evaluation.py${NC}"
else
    echo -e "${YELLOW}Dry run complete. No files were transferred.${NC}"
    echo -e "${YELLOW}Run without --dry-run to actually download files.${NC}"
fi
