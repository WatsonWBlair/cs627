#!/bin/bash
# ==============================================================================
# Setup AWS Resources for CS627 Token-Based Training
# ==============================================================================
# This script creates all necessary AWS resources for the CS627 project:
# - IAM role and instance profile for EC2-S3 access
# - S3 bucket with proper structure and lifecycle policies
# - Security group with appropriate rules
# - Outputs configuration to .env.cloud file
#
# Usage:
#   ./scripts/setup_aws_resources.sh [--region REGION] [--bucket-suffix SUFFIX]
#
# Example:
#   ./scripts/setup_aws_resources.sh --region us-east-1 --bucket-suffix dev
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
REGION="${AWS_REGION:-us-east-1}"
BUCKET_SUFFIX="${BUCKET_SUFFIX:-$(date +%Y%m%d)}"
ROLE_NAME="cs627-ec2-role"
PROFILE_NAME="cs627-profile"
SECURITY_GROUP_NAME="cs627-training"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            REGION="$2"
            shift 2
            ;;
        --bucket-suffix)
            BUCKET_SUFFIX="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--region REGION] [--bucket-suffix SUFFIX]"
            echo "Creates AWS resources for CS627 training pipeline"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set bucket name
S3_BUCKET="cs627-training-${BUCKET_SUFFIX}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  Setting up AWS Resources for CS627                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Region: ${GREEN}$REGION${NC}"
echo -e "  S3 Bucket: ${GREEN}$S3_BUCKET${NC}"
echo -e "  IAM Role: ${GREEN}$ROLE_NAME${NC}"
echo -e "  Instance Profile: ${GREEN}$PROFILE_NAME${NC}"
echo -e "  Security Group: ${GREEN}$SECURITY_GROUP_NAME${NC}"
echo ""

# Step 1: Create IAM Role
echo -e "${BLUE}[1/5] Creating IAM Role...${NC}"

# Check if role exists
if aws iam get-role --role-name $ROLE_NAME --region $REGION &>/dev/null; then
    echo -e "${YELLOW}[SKIP] IAM role $ROLE_NAME already exists${NC}"
else
    # Create trust policy
    cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create role
    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file:///tmp/trust-policy.json \
        --region $REGION

    # Attach S3 policy
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
        --region $REGION

    # Attach CloudWatch policy for monitoring
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy \
        --region $REGION

    echo -e "${GREEN}[OK] IAM role created${NC}"
fi

# Step 2: Create Instance Profile
echo -e "${BLUE}[2/5] Creating Instance Profile...${NC}"

# Check if profile exists
if aws iam get-instance-profile --instance-profile-name $PROFILE_NAME --region $REGION &>/dev/null; then
    echo -e "${YELLOW}[SKIP] Instance profile $PROFILE_NAME already exists${NC}"
else
    # Create instance profile
    aws iam create-instance-profile \
        --instance-profile-name $PROFILE_NAME \
        --region $REGION

    # Add role to profile
    aws iam add-role-to-instance-profile \
        --instance-profile-name $PROFILE_NAME \
        --role-name $ROLE_NAME \
        --region $REGION

    echo -e "${GREEN}[OK] Instance profile created${NC}"
fi

# Step 3: Create S3 Bucket
echo -e "${BLUE}[3/5] Creating S3 Bucket...${NC}"

# Check if bucket exists
if aws s3api head-bucket --bucket $S3_BUCKET --region $REGION &>/dev/null; then
    echo -e "${YELLOW}[SKIP] S3 bucket $S3_BUCKET already exists${NC}"
else
    # Create bucket
    if [ "$REGION" = "us-east-1" ]; then
        aws s3 mb s3://$S3_BUCKET --region $REGION
    else
        aws s3 mb s3://$S3_BUCKET --region $REGION --create-bucket-configuration LocationConstraint=$REGION
    fi

    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket $S3_BUCKET \
        --versioning-configuration Status=Enabled \
        --region $REGION

    # Create folder structure
    aws s3api put-object --bucket $S3_BUCKET --key tokens/ --region $REGION
    aws s3api put-object --bucket $S3_BUCKET --key weights/ --region $REGION
    aws s3api put-object --bucket $S3_BUCKET --key results/ --region $REGION
    aws s3api put-object --bucket $S3_BUCKET --key logs/ --region $REGION
    aws s3api put-object --bucket $S3_BUCKET --key datasets/ --region $REGION

    # Set lifecycle policy
    cat > /tmp/lifecycle.json <<EOF
{
  "Rules": [
    {
      "Id": "TransitionOldTokens",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER_IR"
        }
      ],
      "Filter": {
        "Prefix": "tokens/"
      }
    },
    {
      "Id": "DeleteOldLogs",
      "Status": "Enabled",
      "Expiration": {
        "Days": 30
      },
      "Filter": {
        "Prefix": "logs/"
      }
    }
  ]
}
EOF

    aws s3api put-bucket-lifecycle-configuration \
        --bucket $S3_BUCKET \
        --lifecycle-configuration file:///tmp/lifecycle.json \
        --region $REGION

    echo -e "${GREEN}[OK] S3 bucket created with lifecycle policies${NC}"
fi

# Step 4: Create Security Group
echo -e "${BLUE}[4/5] Creating Security Group...${NC}"

# Get VPC ID
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $REGION)

# Check if security group exists
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query "SecurityGroups[0].GroupId" \
    --output text \
    --region $REGION 2>/dev/null || echo "")

if [ "$SG_ID" != "" ] && [ "$SG_ID" != "None" ]; then
    echo -e "${YELLOW}[SKIP] Security group $SECURITY_GROUP_NAME already exists (ID: $SG_ID)${NC}"
else
    # Create security group
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for CS627 training instances" \
        --vpc-id $VPC_ID \
        --query "GroupId" \
        --output text \
        --region $REGION)

    # Add SSH rule
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    # Add Jupyter rule (optional)
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8888 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    # Add TensorBoard rule (optional)
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 6006 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    echo -e "${GREEN}[OK] Security group created (ID: $SG_ID)${NC}"
fi

# Step 5: Create .env.cloud file
echo -e "${BLUE}[5/5] Creating .env.cloud configuration file...${NC}"

cat > .env.cloud <<EOF
# AWS Configuration for CS627 Training
# Generated on $(date)

# Region Configuration
AWS_DEFAULT_REGION=$REGION

# S3 Configuration
S3_BUCKET=$S3_BUCKET
S3_TOKEN_PATH=s3://$S3_BUCKET/tokens/
S3_WEIGHTS_PATH=s3://$S3_BUCKET/weights/
S3_RESULTS_PATH=s3://$S3_BUCKET/results/
S3_LOGS_PATH=s3://$S3_BUCKET/logs/

# IAM Configuration
IAM_ROLE=$ROLE_NAME
INSTANCE_PROFILE=$PROFILE_NAME

# Security Configuration
SECURITY_GROUP_ID=$SG_ID
SECURITY_GROUP_NAME=$SECURITY_GROUP_NAME

# Instance Configuration (update as needed)
INSTANCE_TYPE=g5.4xlarge
KEY_NAME=your-key-name  # UPDATE THIS

# Training Configuration
BATCH_SIZE=32
USE_AMP=1
CACHE_SIZE=2000
EOF

echo -e "${GREEN}[OK] Configuration saved to .env.cloud${NC}"

# Cleanup temporary files
rm -f /tmp/trust-policy.json /tmp/lifecycle.json

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    AWS Resources Setup Complete!                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Resources Created:${NC}"
echo -e "  ✓ IAM Role: ${GREEN}$ROLE_NAME${NC}"
echo -e "  ✓ Instance Profile: ${GREEN}$PROFILE_NAME${NC}"
echo -e "  ✓ S3 Bucket: ${GREEN}$S3_BUCKET${NC}"
echo -e "  ✓ Security Group: ${GREEN}$SG_ID${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Update KEY_NAME in .env.cloud with your SSH key pair name"
echo "2. Source the environment: source .env.cloud"
echo "3. Launch an instance using: ./scripts/cloud/launch_training_instance.sh"
echo ""
echo -e "${YELLOW}To delete all resources later, run:${NC}"
echo "  ./scripts/cleanup_aws_resources.sh"