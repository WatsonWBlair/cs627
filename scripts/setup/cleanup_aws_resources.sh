#!/bin/bash
# ==============================================================================
# Cleanup AWS Resources for CS627 Project
# ==============================================================================
# This script safely removes AWS resources created for the CS627 project.
# It will prompt for confirmation before deleting resources.
#
# Usage:
#   ./scripts/cleanup_aws_resources.sh [--force] [--keep-bucket]
#
# Options:
#   --force       Skip confirmation prompts
#   --keep-bucket Keep S3 bucket and its contents
#   --help        Show this help message
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - read from .env.cloud if it exists
if [ -f ".env.cloud" ]; then
    source .env.cloud
else
    echo -e "${YELLOW}[WARN] .env.cloud not found. Using default values.${NC}"
    REGION="${AWS_DEFAULT_REGION:-us-east-1}"
    S3_BUCKET="${S3_BUCKET:-}"
    ROLE_NAME="${IAM_ROLE:-cs627-ec2-role}"
    PROFILE_NAME="${INSTANCE_PROFILE:-cs627-profile}"
    SECURITY_GROUP_NAME="${SECURITY_GROUP_NAME:-cs627-training}"
fi

# Parse command line arguments
FORCE=false
KEEP_BUCKET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --keep-bucket)
            KEEP_BUCKET=true
            shift
            ;;
        --help)
            head -n 14 "$0" | tail -n 10
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  Cleanup AWS Resources for CS627                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Resources to be removed:${NC}"
echo -e "  • IAM Role: ${RED}$ROLE_NAME${NC}"
echo -e "  • Instance Profile: ${RED}$PROFILE_NAME${NC}"
if [ "$KEEP_BUCKET" = false ] && [ -n "$S3_BUCKET" ]; then
    echo -e "  • S3 Bucket: ${RED}$S3_BUCKET${NC} (and all contents)"
fi
echo -e "  • Security Group: ${RED}$SECURITY_GROUP_NAME${NC}"
echo ""

# Check for running instances
echo -e "${BLUE}Checking for running instances...${NC}"
RUNNING_INSTANCES=$(aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" "Name=tag:Project,Values=cs627-svs" \
    --query "Reservations[].Instances[].InstanceId" \
    --output text \
    --region $REGION 2>/dev/null || echo "")

if [ -n "$RUNNING_INSTANCES" ]; then
    echo -e "${RED}[WARN] Found running CS627 instances: $RUNNING_INSTANCES${NC}"
    echo -e "${YELLOW}Please terminate these instances first to avoid additional charges.${NC}"
    if [ "$FORCE" = false ]; then
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cleanup cancelled."
            exit 1
        fi
    fi
fi

# Confirmation
if [ "$FORCE" = false ]; then
    echo -e "${RED}⚠️  WARNING: This will permanently delete the resources listed above.${NC}"
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

# Step 1: Remove Instance Profile
echo -e "${BLUE}[1/4] Removing Instance Profile...${NC}"
if aws iam get-instance-profile --instance-profile-name $PROFILE_NAME --region $REGION &>/dev/null; then
    # Remove role from instance profile
    aws iam remove-role-from-instance-profile \
        --instance-profile-name $PROFILE_NAME \
        --role-name $ROLE_NAME \
        --region $REGION 2>/dev/null || true
    
    # Delete instance profile
    aws iam delete-instance-profile \
        --instance-profile-name $PROFILE_NAME \
        --region $REGION
    
    echo -e "${GREEN}[OK] Instance profile removed${NC}"
else
    echo -e "${YELLOW}[SKIP] Instance profile not found${NC}"
fi

# Step 2: Remove IAM Role
echo -e "${BLUE}[2/4] Removing IAM Role...${NC}"
if aws iam get-role --role-name $ROLE_NAME --region $REGION &>/dev/null; then
    # Detach policies
    aws iam detach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
        --region $REGION 2>/dev/null || true
    
    aws iam detach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
        --region $REGION 2>/dev/null || true
    
    # Delete role
    aws iam delete-role \
        --role-name $ROLE_NAME \
        --region $REGION
    
    echo -e "${GREEN}[OK] IAM role removed${NC}"
else
    echo -e "${YELLOW}[SKIP] IAM role not found${NC}"
fi

# Step 3: Delete S3 Bucket
if [ "$KEEP_BUCKET" = false ] && [ -n "$S3_BUCKET" ]; then
    echo -e "${BLUE}[3/4] Removing S3 Bucket...${NC}"
    
    # Check if bucket exists
    if aws s3api head-bucket --bucket $S3_BUCKET --region $REGION &>/dev/null; then
        # Get bucket size
        BUCKET_SIZE=$(aws s3 ls s3://$S3_BUCKET --recursive --summarize --region $REGION | grep "Total Size" | cut -d: -f2 | xargs)
        if [ -n "$BUCKET_SIZE" ] && [ "$BUCKET_SIZE" != "0" ]; then
            echo -e "${YELLOW}Bucket contains data (size: $BUCKET_SIZE bytes)${NC}"
            if [ "$FORCE" = false ]; then
                read -p "Delete all bucket contents? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo -e "${YELLOW}[SKIP] Keeping S3 bucket${NC}"
                    KEEP_BUCKET=true
                fi
            fi
        fi
        
        if [ "$KEEP_BUCKET" = false ]; then
            # Remove all objects and versions
            echo "Emptying bucket..."
            aws s3 rm s3://$S3_BUCKET --recursive --region $REGION
            
            # Remove all object versions (if versioning was enabled)
            aws s3api list-object-versions --bucket $S3_BUCKET --region $REGION \
                --query 'Versions[].{Key:Key,VersionId:VersionId}' \
                --output json 2>/dev/null | \
                jq -r '.[] | "--key '\''\(.Key)'\'' --version-id \(.VersionId)"' | \
                while read -r line; do
                    eval aws s3api delete-object --bucket $S3_BUCKET $line --region $REGION 2>/dev/null || true
                done
            
            # Remove all delete markers
            aws s3api list-object-versions --bucket $S3_BUCKET --region $REGION \
                --query 'DeleteMarkers[].{Key:Key,VersionId:VersionId}' \
                --output json 2>/dev/null | \
                jq -r '.[] | "--key '\''\(.Key)'\'' --version-id \(.VersionId)"' | \
                while read -r line; do
                    eval aws s3api delete-object --bucket $S3_BUCKET $line --region $REGION 2>/dev/null || true
                done
            
            # Delete bucket
            aws s3 rb s3://$S3_BUCKET --force --region $REGION
            
            echo -e "${GREEN}[OK] S3 bucket removed${NC}"
        fi
    else
        echo -e "${YELLOW}[SKIP] S3 bucket not found${NC}"
    fi
else
    echo -e "${YELLOW}[3/4] Skipping S3 bucket removal${NC}"
fi

# Step 4: Delete Security Group
echo -e "${BLUE}[4/4] Removing Security Group...${NC}"

# Get security group ID
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query "SecurityGroups[0].GroupId" \
    --output text \
    --region $REGION 2>/dev/null || echo "None")

if [ "$SG_ID" != "None" ] && [ -n "$SG_ID" ]; then
    # Check if security group is in use
    IN_USE=$(aws ec2 describe-instances \
        --filters "Name=instance.group-id,Values=$SG_ID" "Name=instance-state-name,Values=running,stopped" \
        --query "Reservations[].Instances[].InstanceId" \
        --output text \
        --region $REGION 2>/dev/null || echo "")
    
    if [ -n "$IN_USE" ]; then
        echo -e "${YELLOW}[SKIP] Security group is in use by instances: $IN_USE${NC}"
    else
        aws ec2 delete-security-group \
            --group-id $SG_ID \
            --region $REGION
        
        echo -e "${GREEN}[OK] Security group removed${NC}"
    fi
else
    echo -e "${YELLOW}[SKIP] Security group not found${NC}"
fi

# Remove .env.cloud file
if [ -f ".env.cloud" ] && [ "$FORCE" = false ]; then
    read -p "Remove .env.cloud configuration file? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm .env.cloud
        echo -e "${GREEN}[OK] .env.cloud removed${NC}"
    fi
elif [ -f ".env.cloud" ] && [ "$FORCE" = true ]; then
    rm .env.cloud
    echo -e "${GREEN}[OK] .env.cloud removed${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                     AWS Resources Cleanup Complete                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$KEEP_BUCKET" = true ] && [ -n "$S3_BUCKET" ]; then
    echo -e "${YELLOW}Note: S3 bucket $S3_BUCKET was kept.${NC}"
    echo -e "To remove it later, run: aws s3 rb s3://$S3_BUCKET --force"
    echo ""
fi

echo -e "${GREEN}All specified AWS resources have been removed.${NC}"