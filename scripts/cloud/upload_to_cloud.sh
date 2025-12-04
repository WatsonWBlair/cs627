#!/bin/bash
# ==============================================================================
# Upload preprocessed data to cloud instance for training
# ==============================================================================
# This script uploads:
#   - data/cmumosi/ (preprocessed audio segments and video frames ~500MB)
#   - AdapterWeights/ (optional - if resuming training)
#
# Usage:
#   ./scripts/cloud/upload_to_cloud.sh [OPTIONS]
#
# Options:
#   --data-only       Upload only preprocessed data (default: data + weights)
#   --weights-only    Upload only adapter weights
#   --skip-weights    Upload data but skip adapter weights
#   --dry-run         Show what would be transferred without actually transferring
#   --help            Show this help message
#
# Recommended workflow:
#   1. Preprocess data locally (download videos + extract segments)
#   2. Upload to cloud with this script (~500MB vs 10GB)
#   3. Train on GPU instance
#   4. Download results with download_from_cloud.sh
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
MOSI_DATA_PATH="${MOSI_DATA_PATH:-data/cmumosi}"
ADAPTER_WEIGHTS_DIR="${ADAPTER_WEIGHTS_DIR:-AdapterWeights}"
CLOUD_SSH_KEY="${CLOUD_SSH_KEY}"
CLOUD_PORT="${CLOUD_PORT:-22}"

# Parse command line arguments
UPLOAD_DATA=true
UPLOAD_WEIGHTS=true
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            UPLOAD_WEIGHTS=false
            shift
            ;;
        --weights-only)
            UPLOAD_DATA=false
            shift
            ;;
        --skip-weights)
            UPLOAD_WEIGHTS=false
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --help)
            head -n 27 "$0" | tail -n 24
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
echo -e "${BLUE}║                    Upload to Cloud Instance                           ║${NC}"
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
# Validate local data exists
# ==============================================================================

if [ "$UPLOAD_DATA" = true ]; then
    if [ ! -d "$PROJECT_ROOT/$MOSI_DATA_PATH" ]; then
        echo -e "${RED}Error: Local data directory not found: $MOSI_DATA_PATH${NC}"
        echo -e "${YELLOW}Please preprocess data locally first:${NC}"
        echo -e "  python scripts/data_wrangling/download_all_mosi_videos.py"
        echo -e "  python scripts/data_wrangling/extract_all_segments.py"
        exit 1
    fi
fi

if [ "$UPLOAD_WEIGHTS" = true ]; then
    if [ ! -d "$PROJECT_ROOT/$ADAPTER_WEIGHTS_DIR" ]; then
        echo -e "${YELLOW}⚠️  Warning: AdapterWeights directory not found${NC}"
        echo -e "${YELLOW}Skipping weights upload (this is normal for new training runs)${NC}"
        UPLOAD_WEIGHTS=false
        echo ""
    fi
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
# Create remote directories
# ==============================================================================

if [ -z "$DRY_RUN" ]; then
    echo -e "${BLUE}[2/3] Creating remote directories...${NC}"
    ssh $SSH_OPTS "$CLOUD_CONN" "mkdir -p $CLOUD_PROJECT_DIR/$MOSI_DATA_PATH"
    if [ "$UPLOAD_WEIGHTS" = true ]; then
        ssh $SSH_OPTS "$CLOUD_CONN" "mkdir -p $CLOUD_PROJECT_DIR/$ADAPTER_WEIGHTS_DIR"
    fi
    echo -e "${GREEN}✓ Remote directories created${NC}"
    echo ""
else
    echo -e "${BLUE}[2/3] Skipping directory creation (dry run)${NC}"
    echo ""
fi

# ==============================================================================
# Upload preprocessed data
# ==============================================================================

if [ "$UPLOAD_DATA" = true ]; then
    echo -e "${BLUE}[3/3] Uploading preprocessed data (~500MB)...${NC}"
    echo -e "${YELLOW}This may take 5-10 minutes depending on your connection${NC}"
    echo ""

    eval rsync $RSYNC_OPTS \
        "$PROJECT_ROOT/$MOSI_DATA_PATH/" \
        "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$MOSI_DATA_PATH/"

    if [ -z "$DRY_RUN" ]; then
        echo ""
        echo -e "${GREEN}✓ Data uploaded successfully${NC}"
    fi
    echo ""
else
    echo -e "${BLUE}[3/3] Skipping data upload (--weights-only specified)${NC}"
    echo ""
fi

# ==============================================================================
# Upload adapter weights (if resuming training)
# ==============================================================================

if [ "$UPLOAD_WEIGHTS" = true ]; then
    echo -e "${BLUE}[Bonus] Uploading adapter weights...${NC}"

    eval rsync $RSYNC_OPTS \
        "$PROJECT_ROOT/$ADAPTER_WEIGHTS_DIR/" \
        "$CLOUD_CONN:$CLOUD_PROJECT_DIR/$ADAPTER_WEIGHTS_DIR/"

    if [ -z "$DRY_RUN" ]; then
        echo -e "${GREEN}✓ Adapter weights uploaded${NC}"
    fi
    echo ""
fi

# ==============================================================================
# Summary
# ==============================================================================

if [ -z "$DRY_RUN" ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                        Upload Complete!                                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Uploaded artifacts:${NC}"

    if [ "$UPLOAD_DATA" = true ]; then
        echo -e "  📦 Preprocessed data: ${GREEN}$MOSI_DATA_PATH/${NC}"
    fi

    if [ "$UPLOAD_WEIGHTS" = true ]; then
        echo -e "  🎯 Adapter weights: ${GREEN}$ADAPTER_WEIGHTS_DIR/${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Next steps on cloud instance:${NC}"
    echo -e "  1. SSH into instance: ${BLUE}ssh $CLOUD_CONN${NC}"
    echo -e "  2. Navigate to project: ${BLUE}cd $CLOUD_PROJECT_DIR${NC}"
    echo -e "  3. Install dependencies: ${BLUE}pip install -r requirements.txt${NC}"
    echo -e "  4. Start training: ${BLUE}python src/Training/train_encoders.py${NC}"
    echo ""
    echo -e "${YELLOW}After training completes:${NC}"
    echo -e "  Download results: ${BLUE}./scripts/cloud/download_from_cloud.sh${NC}"
else
    echo -e "${YELLOW}Dry run complete. No files were transferred.${NC}"
    echo -e "${YELLOW}Run without --dry-run to actually upload files.${NC}"
fi
