#!/bin/bash
# ==============================================================================
# Launch AWS GPU Instance for Training
# ==============================================================================
# This script launches an on-demand GPU instance optimized for fast training.
#
# Default: g5.4xlarge (A10G, 24GB) - Best value for 100 epoch training
#
# Usage:
#   ./scripts/cloud/launch_training_instance.sh [OPTIONS]
#
# Options:
#   --instance-type TYPE    Instance type (default: g5.4xlarge)
#   --region REGION         AWS region (default: us-east-1)
#   --key-name NAME         SSH key pair name (required)
#   --security-group ID     Security group ID (required)
#   --spot                  Use spot instance (not recommended for <4hr training)
#   --dry-run               Show what would be launched without launching
#   --help                  Show this help message
#
# Examples:
#   # Launch with defaults (on-demand g5.4xlarge)
#   ./scripts/cloud/launch_training_instance.sh --key-name my-key --security-group sg-xxx
#
#   # Launch p3.2xlarge (V100) for fastest training
#   ./scripts/cloud/launch_training_instance.sh --instance-type p3.2xlarge --key-name my-key --security-group sg-xxx
#
#   # Launch with spot (not recommended for short training)
#   ./scripts/cloud/launch_training_instance.sh --spot --key-name my-key --security-group sg-xxx
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
INSTANCE_TYPE="g5.4xlarge"
REGION="us-east-1"
AMI_ID="ami-0c7217cdde317cfec"  # Deep Learning AMI (Ubuntu 20.04) - US East 1
KEY_NAME=""
SECURITY_GROUP=""
USE_SPOT=false
DRY_RUN=""
SUBNET_ID=""
IAM_ROLE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --key-name)
            KEY_NAME="$2"
            shift 2
            ;;
        --security-group)
            SECURITY_GROUP="$2"
            shift 2
            ;;
        --subnet-id)
            SUBNET_ID="$2"
            shift 2
            ;;
        --iam-role)
            IAM_ROLE="$2"
            shift 2
            ;;
        --spot)
            USE_SPOT=true
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --help)
            head -n 29 "$0" | tail -n 26
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$KEY_NAME" ]; then
    echo -e "${RED}Error: --key-name is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$SECURITY_GROUP" ]; then
    echo -e "${RED}Error: --security-group is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Get AMI ID for the specified region (if not us-east-1)
if [ "$REGION" != "us-east-1" ]; then
    echo -e "${YELLOW}⚠️  Using custom region. You may need to specify a different AMI ID.${NC}"
fi

# Instance type configurations
declare -A INSTANCE_INFO
INSTANCE_INFO["g5.4xlarge"]="A10G 24GB|~2.5-3hr|$1.63/hr|Recommended"
INSTANCE_INFO["g5.8xlarge"]="A10G 24GB|~2-3hr|$2.45/hr|High Memory"
INSTANCE_INFO["p3.2xlarge"]="V100 16GB|~2-2.5hr|$3.06/hr|Fastest"
INSTANCE_INFO["g4dn.xlarge"]="T4 16GB|~5-7hr|$0.526/hr|Budget"
INSTANCE_INFO["g4dn.2xlarge"]="T4 16GB|~5-7hr|$0.752/hr|Budget+"

# Display launch configuration
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Launch AWS Training Instance                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Instance Type: ${GREEN}$INSTANCE_TYPE${NC}"

if [ -n "${INSTANCE_INFO[$INSTANCE_TYPE]}" ]; then
    IFS='|' read -r gpu time cost note <<< "${INSTANCE_INFO[$INSTANCE_TYPE]}"
    echo -e "  GPU: ${GREEN}$gpu${NC}"
    echo -e "  Expected Time (100 epochs): ${GREEN}$time${NC}"
    echo -e "  Cost: ${GREEN}$cost${NC} on-demand"
    echo -e "  Note: ${GREEN}$note${NC}"
fi

echo -e "  Region: ${GREEN}$REGION${NC}"
echo -e "  Key Name: ${GREEN}$KEY_NAME${NC}"
echo -e "  Security Group: ${GREEN}$SECURITY_GROUP${NC}"
echo -e "  Purchase Option: ${GREEN}$([ "$USE_SPOT" = true ] && echo "Spot Instance ⚠️" || echo "On-Demand ✓")${NC}"

if [ -n "$DRY_RUN" ]; then
    echo -e "  ${YELLOW}⚠️  DRY RUN MODE - No instance will be launched${NC}"
fi

echo ""

# Warning for spot instances
if [ "$USE_SPOT" = true ]; then
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║                    ⚠️  SPOT INSTANCE WARNING ⚠️                         ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${YELLOW}Spot instances can be interrupted with 2 minutes notice!${NC}"
    echo -e "${YELLOW}For training runs <4 hours, on-demand is recommended.${NC}"
    echo ""
    read -p "Continue with spot instance? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Launch cancelled."
        exit 0
    fi
fi

# Build launch command
echo -e "${BLUE}[1/4] Preparing launch configuration...${NC}"

# User data script (runs on instance startup)
USER_DATA=$(cat <<'EOF'
#!/bin/bash
# Instance initialization script

# Update system
apt-get update

# Install essential tools
apt-get install -y git wget curl htop nvtop

# Configure git
git config --global credential.helper store

# Create training directory
mkdir -p /home/ubuntu/cs627

# Print GPU info
nvidia-smi

echo "Instance initialized and ready for training!"
echo "SSH into the instance and run:"
echo "  cd cs627"
echo "  git clone https://github.com/WatsonWBlair/cs627.git ."
echo "  pip install -r requirements.txt"
echo "  python src/Training/train_encoders.py"
EOF
)

# Build AWS CLI command
LAUNCH_CMD="aws ec2 run-instances"
LAUNCH_CMD="$LAUNCH_CMD --region $REGION"
LAUNCH_CMD="$LAUNCH_CMD --image-id $AMI_ID"
LAUNCH_CMD="$LAUNCH_CMD --instance-type $INSTANCE_TYPE"
LAUNCH_CMD="$LAUNCH_CMD --key-name $KEY_NAME"
LAUNCH_CMD="$LAUNCH_CMD --security-group-ids $SECURITY_GROUP"
LAUNCH_CMD="$LAUNCH_CMD --user-data \"$USER_DATA\""

if [ -n "$SUBNET_ID" ]; then
    LAUNCH_CMD="$LAUNCH_CMD --subnet-id $SUBNET_ID"
fi

if [ -n "$IAM_ROLE" ]; then
    LAUNCH_CMD="$LAUNCH_CMD --iam-instance-profile Name=$IAM_ROLE"
fi

# Block device mapping (increase root volume to 100GB)
LAUNCH_CMD="$LAUNCH_CMD --block-device-mappings"
LAUNCH_CMD="$LAUNCH_CMD 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}'"

# Spot instance configuration
if [ "$USE_SPOT" = true ]; then
    LAUNCH_CMD="$LAUNCH_CMD --instance-market-options"
    LAUNCH_CMD="$LAUNCH_CMD 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'"
fi

# Tags
LAUNCH_CMD="$LAUNCH_CMD --tag-specifications"
LAUNCH_CMD="$LAUNCH_CMD 'ResourceType=instance,Tags=[{Key=Name,Value=cs627-training},{Key=Project,Value=cs627-svs},{Key=Purpose,Value=encoder-training}]'"

if [ -n "$DRY_RUN" ]; then
    LAUNCH_CMD="$LAUNCH_CMD $DRY_RUN"
fi

echo -e "${GREEN}✓ Configuration prepared${NC}"
echo ""

# Launch instance
echo -e "${BLUE}[2/4] Launching instance...${NC}"

# Execute launch command
LAUNCH_OUTPUT=$(eval $LAUNCH_CMD 2>&1)
LAUNCH_EXIT_CODE=$?

if [ $LAUNCH_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}✗ Failed to launch instance${NC}"
    echo "$LAUNCH_OUTPUT"
    exit 1
fi

if [ -n "$DRY_RUN" ]; then
    echo -e "${YELLOW}Dry run successful. No instance was launched.${NC}"
    exit 0
fi

# Extract instance ID
INSTANCE_ID=$(echo "$LAUNCH_OUTPUT" | grep -o '"InstanceId": "[^"]*"' | cut -d'"' -f4)

if [ -z "$INSTANCE_ID" ]; then
    echo -e "${RED}✗ Could not extract instance ID from response${NC}"
    echo "$LAUNCH_OUTPUT"
    exit 1
fi

echo -e "${GREEN}✓ Instance launched: $INSTANCE_ID${NC}"
echo ""

# Wait for instance to be running
echo -e "${BLUE}[3/4] Waiting for instance to start...${NC}"
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Get instance details
INSTANCE_DETAILS=$(aws ec2 describe-instances --region $REGION --instance-ids $INSTANCE_ID)
PUBLIC_IP=$(echo "$INSTANCE_DETAILS" | grep -o '"PublicIpAddress": "[^"]*"' | cut -d'"' -f4)
PRIVATE_IP=$(echo "$INSTANCE_DETAILS" | grep -o '"PrivateIpAddress": "[^"]*"' | cut -d'"' -f4)

echo -e "${GREEN}✓ Instance is running${NC}"
echo ""

# Display connection info
echo -e "${BLUE}[4/4] Instance ready!${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Instance Successfully Launched!                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Instance Details:${NC}"
echo -e "  Instance ID: ${GREEN}$INSTANCE_ID${NC}"
echo -e "  Instance Type: ${GREEN}$INSTANCE_TYPE${NC}"
echo -e "  Public IP: ${GREEN}$PUBLIC_IP${NC}"
echo -e "  Private IP: ${GREEN}$PRIVATE_IP${NC}"
echo -e "  Region: ${GREEN}$REGION${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo -e "${BLUE}1. Wait ~2 minutes for instance initialization${NC}"
echo ""
echo -e "${BLUE}2. SSH into the instance:${NC}"
echo -e "   ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP"
echo ""
echo -e "${BLUE}3. Set up the project:${NC}"
echo -e "   cd /home/ubuntu"
echo -e "   git clone https://github.com/WatsonWBlair/cs627.git"
echo -e "   cd cs627"
echo ""
echo -e "${BLUE}4. Configure cloud credentials (on local machine):${NC}"
echo -e "   cp .env.cloud.example .env.cloud"
echo -e "   nano .env.cloud  # Update CLOUD_HOST=$PUBLIC_IP"
echo ""
echo -e "${BLUE}5. Upload preprocessed data:${NC}"
echo -e "   ./scripts/cloud/upload_to_cloud.sh"
echo ""
echo -e "${BLUE}6. Start training (on instance):${NC}"
echo -e "   python src/Training/train_encoders.py"
echo ""
echo -e "${BLUE}7. Download results (after training):${NC}"
echo -e "   ./scripts/cloud/download_from_cloud.sh"
echo ""
echo -e "${BLUE}8. IMPORTANT: Terminate instance when done:${NC}"
echo -e "   aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
echo ""
echo -e "${YELLOW}Expected Training Cost:${NC}"
if [ -n "${INSTANCE_INFO[$INSTANCE_TYPE]}" ]; then
    IFS='|' read -r gpu time cost note <<< "${INSTANCE_INFO[$INSTANCE_TYPE]}"
    echo -e "  Duration: ${GREEN}$time${NC}"
    echo -e "  Hourly Rate: ${GREEN}$cost${NC}"
    # Calculate approximate cost (mid-range estimate)
    if [[ "$time" =~ ([0-9.]+)-([0-9.]+)hr ]]; then
        LOW_HOURS="${BASH_REMATCH[1]}"
        HIGH_HOURS="${BASH_REMATCH[2]}"
        if [[ "$cost" =~ \$([0-9.]+) ]]; then
            RATE="${BASH_REMATCH[1]}"
            LOW_COST=$(echo "$LOW_HOURS * $RATE" | bc -l)
            HIGH_COST=$(echo "$HIGH_HOURS * $RATE" | bc -l)
            echo -e "  Total Cost: ${GREEN}\$$(printf '%.2f' $LOW_COST)-\$$(printf '%.2f' $HIGH_COST)${NC}"
        fi
    fi
fi
echo ""
echo -e "${RED}⚠️  Remember to terminate the instance to avoid charges!${NC}"
echo ""

# Save instance info to file
INFO_FILE=".cloud_instance_info"
cat > "$INFO_FILE" <<EOL
INSTANCE_ID=$INSTANCE_ID
INSTANCE_TYPE=$INSTANCE_TYPE
PUBLIC_IP=$PUBLIC_IP
PRIVATE_IP=$PRIVATE_IP
REGION=$REGION
LAUNCHED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOL

echo -e "${GREEN}Instance info saved to: $INFO_FILE${NC}"
