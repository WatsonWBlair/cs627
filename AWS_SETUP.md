# AWS Setup Guide

Deploy CS627 training on AWS EC2 with GPU support.

## Prerequisites: IAM and S3 Setup

### 1. Create IAM Role for EC2

```bash
# Create trust policy file
cat > trust-policy.json <<EOF
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

# Create IAM role
aws iam create-role \
  --role-name cs627-ec2-role \
  --assume-role-policy-document file://trust-policy.json

# Attach S3 access policy
aws iam attach-role-policy \
  --role-name cs627-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name cs627-profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name cs627-profile \
  --role-name cs627-ec2-role
```

### 2. Create S3 Bucket

```bash
# Set your unique bucket name
export S3_BUCKET="cs627-training-$(whoami)-$(date +%Y%m%d)"

# Create bucket
aws s3 mb s3://$S3_BUCKET --region us-east-1

# Enable versioning for safety
aws s3api put-bucket-versioning \
  --bucket $S3_BUCKET \
  --versioning-configuration Status=Enabled

# Create folder structure
aws s3api put-object --bucket $S3_BUCKET --key tokens/
aws s3api put-object --bucket $S3_BUCKET --key weights/
aws s3api put-object --bucket $S3_BUCKET --key results/
aws s3api put-object --bucket $S3_BUCKET --key logs/

# Set lifecycle policy to transition old tokens to cheaper storage
cat > lifecycle.json <<EOF
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
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket $S3_BUCKET \
  --lifecycle-configuration file://lifecycle.json
```

### 3. Configure Bucket Policy (Optional - for team sharing)

```bash
cat > bucket-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowTeamAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::ACCOUNT-ID:user/teammate1",
          "arn:aws:iam::ACCOUNT-ID:user/teammate2"
        ]
      },
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::$S3_BUCKET/*",
        "arn:aws:s3:::$S3_BUCKET"
      ]
    }
  ]
}
EOF

# Apply bucket policy (update ACCOUNT-ID and usernames first)
aws s3api put-bucket-policy \
  --bucket $S3_BUCKET \
  --policy file://bucket-policy.json
```

## AWS Console Setup (Web UI)

For users who prefer the AWS Management Console web interface over CLI commands, follow these step-by-step instructions.

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     AWS Resource Setup                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. IAM Role ──────► 2. S3 Bucket                           │
│       │                    │                                 │
│       ▼                    ▼                                 │
│  Instance Profile     Token Storage                          │
│       │                    │                                 │
│       └────────┬───────────┘                                 │
│                ▼                                              │
│         3. Security Group                                    │
│                │                                              │
│                ▼                                              │
│         4. ECR Repository                                    │
│                │                                              │
│                ▼                                              │
│         5. EC2 Instance                                      │
│                │                                              │
│                ▼                                              │
│         6. Training Pipeline                                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: IAM Role Configuration

#### 1.1 Navigate to IAM Service
1. Open [AWS Console](https://console.aws.amazon.com)
2. Search for "IAM" in the search bar
3. Click on **IAM** to open the Identity and Access Management console

#### 1.2 Create Role
1. In the left sidebar, click **Roles**
2. Click **Create role** button (orange button at top right)
3. Select trusted entity:
   - Choose **AWS service**
   - Select **EC2** from the list
   - Click **Next**

#### 1.3 Add Permissions
1. Search for and select these policies:
   - **AmazonS3FullAccess** (for token storage)
   - **CloudWatchAgentServerPolicy** (for monitoring)
2. Click **Next**
3. Role details:
   - **Role name**: `cs627-ec2-role`
   - **Description**: "Role for CS627 training instances to access S3 and CloudWatch"
4. Click **Create role**

#### 1.4 Note the Role ARN
1. Click on the newly created role `cs627-ec2-role`
2. Copy the **Role ARN** (you'll need this later)
   - Format: `arn:aws:iam::ACCOUNT-ID:role/cs627-ec2-role`

### Step 2: S3 Bucket Setup

#### 2.1 Navigate to S3 Service
1. Return to AWS Console home
2. Search for "S3" in the search bar
3. Click on **S3** to open the storage console

#### 2.2 Create Bucket
1. Click **Create bucket** button (orange button)
2. Bucket configuration:
   - **Bucket name**: `cs627-training-[yourname]-[date]` 
     - Example: `cs627-training-smith-20241208`
     - Must be globally unique
   - **AWS Region**: Select your preferred region (e.g., us-east-1)
   - **Object Ownership**: Keep default (ACLs disabled)
3. **Block Public Access settings**:
   - Keep all boxes checked (block all public access)
4. **Bucket Versioning**:
   - Select **Enable** (for data safety)
5. **Tags** (optional but recommended):
   - Add tag: Key=`Project`, Value=`CS627`
   - Add tag: Key=`Purpose`, Value=`Training`
6. Click **Create bucket**

#### 2.3 Create Folder Structure
1. Click on your newly created bucket name
2. Click **Create folder** button
3. Create these folders one by one:
   - `tokens/` - For pre-generated tokens
   - `weights/` - For trained model weights
   - `results/` - For training results
   - `logs/` - For training logs
   - `datasets/` - For raw datasets

#### 2.4 Configure Lifecycle Policy
1. In your bucket, go to **Management** tab
2. Click **Create lifecycle rule**
3. Lifecycle rule configuration:
   - **Lifecycle rule name**: `TransitionOldTokens`
   - **Rule scope**: Select **Limit the scope using filters**
   - **Prefix**: Enter `tokens/`
4. Lifecycle rule actions:
   - Check **Transition current versions of objects between storage classes**
   - Add transitions:
     - After 30 days → Standard-IA
     - After 90 days → Glacier Instant Retrieval
5. Click **Create rule**

### Step 3: Security Group Configuration

#### 3.1 Navigate to EC2 Service
1. Return to AWS Console home
2. Search for "EC2" in the search bar
3. Click on **EC2** to open the compute console

#### 3.2 Create Security Group
1. In the left sidebar under **Network & Security**, click **Security Groups**
2. Click **Create security group** button
3. Basic details:
   - **Security group name**: `cs627-training`
   - **Description**: "Security group for CS627 training instances"
   - **VPC**: Select your default VPC

#### 3.3 Add Inbound Rules
Click **Add rule** for each of these:

| Type | Port Range | Source | Description |
|------|------------|--------|-------------|
| SSH | 22 | My IP | SSH access |
| Custom TCP | 8888 | My IP | Jupyter Notebook |
| Custom TCP | 6006 | My IP | TensorBoard |

Note: "My IP" automatically fills your current IP address

4. Keep **Outbound rules** as default (all traffic allowed)
5. Add tags:
   - Key=`Project`, Value=`CS627`
6. Click **Create security group**
7. Note the **Security Group ID** (format: `sg-xxxxxxxxx`)

### Step 4: Container Registry (ECR) Setup

#### 4.1 Navigate to ECR Service
1. Return to AWS Console home
2. Search for "ECR" in the search bar
3. Click on **Elastic Container Registry**

#### 4.2 Create Repository
1. Click **Create repository** button
2. Repository configuration:
   - **Visibility**: Private
   - **Repository name**: `cs627-svs`
   - **Tag immutability**: Disabled
   - **Scan on push**: Enabled (recommended)
3. Click **Create repository**

#### 4.3 View Push Commands
1. Click on your repository name `cs627-svs`
2. Click **View push commands** button
3. Save these commands for later use when pushing Docker images

Example commands (your region and account ID will differ):
```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com

# Tag your image
docker tag cs627-svs:latest ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com/cs627-svs:latest

# Push to ECR
docker push ACCOUNT-ID.dkr.ecr.us-east-1.amazonaws.com/cs627-svs:latest
```

### Step 5: Launch EC2 Instance

#### 5.1 Navigate to EC2 Dashboard
1. In EC2 console, click **Instances** in the left sidebar
2. Click **Launch instances** button

#### 5.2 Name and Tags
1. **Name**: `cs627-training-instance`
2. Add tags:
   - Key=`Project`, Value=`CS627`
   - Key=`Purpose`, Value=`TokenTraining`

#### 5.3 Choose AMI
1. Click **Browse more AMIs**
2. Search for "Deep Learning AMI GPU PyTorch"
3. Select **AWS Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5**
   - Or similar recent version
4. Click **Select**

#### 5.4 Instance Type
1. Search for or select **g5.4xlarge**
   - 16 vCPUs, 64 GB RAM, 1 x A10G GPU (24GB VRAM)
   - Cost: ~$1.62/hour
2. For budget option: Select **g4dn.xlarge** (~$0.53/hour)

#### 5.5 Key Pair
1. Select existing key pair or create new one:
   - Click **Create new key pair**
   - **Key pair name**: `cs627-training-key`
   - **Key pair type**: RSA
   - **Private key format**: .pem (for Mac/Linux) or .ppk (for Windows/PuTTY)
   - Click **Create key pair** (file will download automatically)

#### 5.6 Network Settings
1. Click **Edit** button
2. **VPC**: Select default VPC
3. **Subnet**: No preference
4. **Auto-assign public IP**: Enable
5. **Security group**: Select existing
   - Choose **cs627-training** (created in Step 3)

#### 5.7 Configure Storage
1. Change root volume:
   - **Size**: 100 GB (reduced from default)
   - **Volume type**: gp3
   - **Delete on termination**: Checked

#### 5.8 Advanced Details
1. Expand **Advanced details** section
2. **IAM instance profile**: Select `cs627-ec2-role`
3. **User data** (optional - paste this script):
```bash
#!/bin/bash
# Update system
apt-get update
apt-get install -y git wget curl htop nvtop

# Create working directory
mkdir -p /home/ubuntu/cs627
cd /home/ubuntu/cs627

# Clone repository
git clone https://github.com/WatsonWBlair/cs627.git .

# Set permissions
chown -R ubuntu:ubuntu /home/ubuntu/cs627

# Print instructions
echo "Instance ready! SSH in and run:" > /home/ubuntu/setup_instructions.txt
echo "cd cs627" >> /home/ubuntu/setup_instructions.txt
echo "python src/Training/pregenerate_tokens.py" >> /home/ubuntu/setup_instructions.txt
echo "python src/Training/train_adapters.py --mode encoder" >> /home/ubuntu/setup_instructions.txt
```

#### 5.9 Review and Launch
1. Review all settings
2. Click **Launch instance**
3. Wait for instance to start (~2-3 minutes)

### Step 6: Post-Launch Configuration

#### 6.1 Connect to Instance
1. In EC2 console, select your instance
2. Click **Connect** button
3. Choose **SSH client** tab
4. Follow the connection instructions:
```bash
chmod 400 cs627-training-key.pem
ssh -i cs627-training-key.pem ubuntu@[PUBLIC-IP]
```

#### 6.2 Configure AWS CLI on Instance
Once connected via SSH:
```bash
# The instance role provides credentials automatically
aws configure set region us-east-1

# Test S3 access
aws s3 ls s3://cs627-training-[your-bucket-name]/
```

#### 6.3 Pull Docker Image from ECR (if using)
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com

# Pull image
docker pull [ACCOUNT-ID].dkr.ecr.us-east-1.amazonaws.com/cs627-svs:latest
```

#### 6.4 Download Pre-generated Tokens from S3
```bash
# If tokens were pre-generated and uploaded
cd ~/cs627
aws s3 cp s3://cs627-training-[your-bucket]/tokens/ data/pregenerated_tokens/ --recursive
```

#### 6.5 Start Training
```bash
# Activate environment
source venv/bin/activate  # or use conda/docker

# If tokens not pre-generated
python src/Training/pregenerate_tokens.py

# Train adapters
python src/Training/train_adapters.py --mode encoder
python src/Training/train_adapters.py --mode decoder

# Upload results to S3
aws s3 cp OptimalWeights/ s3://cs627-training-[your-bucket]/weights/ --recursive
aws s3 cp Results/ s3://cs627-training-[your-bucket]/results/ --recursive
```

### Step 7: Clean Up Resources

#### 7.1 Terminate EC2 Instance
1. In EC2 console, select your instance
2. Click **Instance state** → **Terminate instance**
3. Confirm termination

#### 7.2 Delete S3 Bucket (Optional)
1. In S3 console, select your bucket
2. Click **Empty** to remove all objects
3. Click **Delete** to remove the bucket

#### 7.3 Delete Security Group
1. In EC2 → Security Groups
2. Select `cs627-training`
3. Click **Actions** → **Delete security groups**

#### 7.4 Delete IAM Role
1. In IAM → Roles
2. Search for `cs627-ec2-role`
3. Select and click **Delete**

#### 7.5 Delete ECR Repository
1. In ECR console, select repository
2. Click **Delete**

### Console Setup Tips

1. **Bookmark Important Pages**:
   - EC2 Instances: `https://console.aws.amazon.com/ec2/v2/home#Instances:`
   - S3 Buckets: `https://s3.console.aws.amazon.com/s3/buckets`
   - CloudWatch: `https://console.aws.amazon.com/cloudwatch/`

2. **Use Tags Consistently**:
   - Always tag with Project=CS627
   - Makes resource tracking and cleanup easier

3. **Set Up Billing Alerts**:
   - Go to Billing → Budgets
   - Create budget alert for unexpected charges

4. **Save Configuration**:
   - After setup, go to EC2 → Launch Templates
   - Create template from your instance for quick re-launch

## Quick Start

### 1. Launch Instance

```bash
# Using AWS CLI with IAM profile
aws ec2 run-instances \
  --image-id ami-04f3e35dc85e9423b \
  --instance-type g5.4xlarge \
  --key-name your-key \
  --security-groups default \
  --iam-instance-profile Name=cs627-profile \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}"
```

### 2. Connect and Setup

```bash
# SSH to instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone repository
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627

# Run setup script
./scripts/setup_remote_instance.sh
```

### 3. Train (Two-Phase Approach)

```bash
# Phase 1: Generate tokens (if not pre-generated)
python src/Training/pregenerate_tokens.py  # ~30 min on g5.4xlarge

# Phase 2: Train adapters (fast)
python src/Training/train_adapters.py --mode encoder  # ~10 min
python src/Training/train_adapters.py --mode decoder  # ~10 min

# Or using Docker
docker pull watsonwb/cs627-svs:latest
make tokens  # Generate tokens
make train   # Train adapters
```

## Recommended Configuration

| Component | Specification | Cost |
|-----------|--------------|------|
| **Instance** | g5.4xlarge | $1.62/hour |
| **GPU** | NVIDIA A10G (24GB) | Included |
| **vCPUs** | 16 | Included |
| **Memory** | 64 GB | Included |
| **Storage** | 100 GB gp3 | $8/month |
| **Region** | us-east-1 | - |

**Estimated training cost**: $2-5 for full pipeline (~1 hour total)

## Instance Options

### GPU Instances

| Instance | GPU | VRAM | vCPUs | RAM | Cost/Hour |
|----------|-----|------|-------|-----|-----------|
| **g5.xlarge** | A10G | 24GB | 4 | 16GB | $1.01 |
| **g5.2xlarge** | A10G | 24GB | 8 | 32GB | $1.21 |
| **g5.4xlarge** | A10G | 24GB | 16 | 64GB | $1.62 |
| **g4dn.xlarge** | T4 | 16GB | 4 | 16GB | $0.53 |
| **p3.2xlarge** | V100 | 16GB | 8 | 61GB | $3.06 |

### CPU Instances (for testing)

| Instance | vCPUs | RAM | Cost/Hour |
|----------|-------|-----|-----------|
| **t3.xlarge** | 4 | 16GB | $0.17 |
| **c5.2xlarge** | 8 | 16GB | $0.34 |

## AMI Selection

### Recommended AMI
- **Name**: AWS Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Latest)
- **Note**: AMI IDs change frequently. Use the command below to find the latest.

### Finding Latest AMI
```bash
# Find latest PyTorch GPU AMI
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=*Deep Learning AMI GPU PyTorch*" \
  --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
  --output text

# Alternative: Ubuntu 22.04 with NVIDIA drivers
aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64*" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text
```

## Step-by-Step Setup

### 1. Create Security Group

```bash
# Create security group
aws ec2 create-security-group \
  --group-name cs627-training \
  --description "CS627 training security group"

# Allow SSH
aws ec2 authorize-security-group-ingress \
  --group-name cs627-training \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# Allow Jupyter (optional)
aws ec2 authorize-security-group-ingress \
  --group-name cs627-training \
  --protocol tcp \
  --port 8888 \
  --cidr 0.0.0.0/0
```

### 2. Launch Instance

```bash
aws ec2 run-instances \
  --image-id ami-04f3e35dc85e9423b \
  --instance-type g5.4xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cs627-training}]' \
  --block-device-mappings file://block-device-mappings.json
```

`block-device-mappings.json`:
```json
[
  {
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": 100,
      "VolumeType": "gp3",
      "DeleteOnTermination": true
    }
  }
]
```

### 3. Connect to Instance

```bash
# Get instance IP
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=cs627-training" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

INSTANCE_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

# SSH to instance
ssh -i your-key.pem ubuntu@$INSTANCE_IP
```

### 4. Setup Environment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Clone repository
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627

# Install Docker (if not present)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

# Pull Docker image
docker pull watsonwb/cs627-svs:latest
```

### 5. Transfer Data (Optional)

```bash
# From local machine
scp -i your-key.pem -r data/pregenerated_tokens ubuntu@$INSTANCE_IP:~/cs627/data/

# Or use S3
aws s3 cp s3://your-bucket/tokens.tar.gz .
tar -xzf tokens.tar.gz
```

### 6. Run Training (Token-Based Workflow)

```bash
# Option 1: Generate tokens on instance (if not pre-generated)
python src/Training/pregenerate_tokens.py  # ~30 min

# Option 2: Download pre-generated tokens from S3
aws s3 cp s3://your-bucket/tokens/ data/pregenerated_tokens/ --recursive

# Train adapters (fast - only trains small MLPs)
python src/Training/train_adapters.py --mode encoder  # ~10 min
python src/Training/train_adapters.py --mode decoder  # ~10 min

# Optional: Run ablation study
python src/Experiments/run_ablation.py --config recommended
```

## Memory Configuration for Token Generation

### Requirements by Batch Size

| Batch Size | RAM Required | Swap Space | GPU VRAM | Recommended Instance |
|------------|--------------|------------|----------|---------------------|
| 4 | 8GB | 4GB | 8GB | g4dn.xlarge |
| 8 | 16GB | 8GB | 12GB | g5.xlarge |
| 16 | 32GB | 16GB | 16GB | g5.2xlarge |
| 32 | 64GB | 32GB | 24GB | g5.4xlarge |
| 64 | 128GB | 0GB | 24GB | g5.8xlarge |

### Configure Swap Space (if needed)

```bash
# Create 32GB swap file
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
```

### Environment Variables for Memory Optimization

```bash
# Optimize PyTorch memory usage
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS=4

# Limit batch size based on available memory
export BATCH_SIZE=32  # Adjust based on instance type

# Enable gradient checkpointing for large batches
export USE_GRADIENT_CHECKPOINTING=1
```

## Token-Based Training on AWS

### Decision: Where to Generate Tokens?

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Local Generation** | Free compute, reusable tokens | Slow (2+ hours on CPU), upload time | Development, multiple experiments |
| **Cloud Generation** | Fast (30 min), no upload needed | Costs ~$1, one-time use | Single training run |
| **Pre-staged on S3** | Instant access, shareable | Storage costs, initial setup | Team collaboration |

### Recommended Workflow

#### Option 1: Generate Locally, Train on Cloud (Most Economical)
```bash
# On local machine (one-time)
python src/Training/pregenerate_tokens.py  # 2+ hours on CPU, free

# Upload to S3
aws s3 cp data/pregenerated_tokens/ s3://your-bucket/tokens/ --recursive

# On cloud instance
aws s3 cp s3://your-bucket/tokens/ data/pregenerated_tokens/ --recursive
python src/Training/train_adapters.py --mode encoder  # 10 min
python src/Training/train_adapters.py --mode decoder  # 10 min
```
**Total cloud time: ~20 minutes ($0.50)**

#### Option 2: Full Cloud Pipeline (Fastest)
```bash
# On cloud instance
python src/Training/pregenerate_tokens.py  # 30 min
python src/Training/train_adapters.py --mode encoder  # 10 min
python src/Training/train_adapters.py --mode decoder  # 10 min

# Save tokens for reuse
aws s3 cp data/pregenerated_tokens/ s3://your-bucket/tokens/ --recursive
```
**Total cloud time: ~50 minutes ($1.35)**

### Token Storage Details

| Dataset | Token Size | S3 Cost/Month | Download Time |
|---------|------------|---------------|---------------|
| MOSI | ~2GB | $0.05 | <1 min |
| MOSEI | ~20GB | $0.50 | 5-10 min |
| Custom | Varies | $0.023/GB | ~200MB/s |

## Data Management

### Using S3

```bash
# Upload tokens to S3
aws s3 cp data/pregenerated_tokens s3://your-bucket/tokens/ --recursive

# Download on instance
aws s3 cp s3://your-bucket/tokens/ data/pregenerated_tokens/ --recursive

# Sync results back
aws s3 sync OptimalWeights/ s3://your-bucket/weights/
aws s3 sync Results/ s3://your-bucket/results/
```

### Using EBS Snapshots

```bash
# Create snapshot of trained model
aws ec2 create-snapshot \
  --volume-id vol-xxxxx \
  --description "CS627 trained models"

# Restore from snapshot
aws ec2 create-volume \
  --snapshot-id snap-xxxxx \
  --availability-zone us-east-1a
```

## Cost Optimization

### Spot Instances (70% savings)

```bash
# Request spot instance
aws ec2 request-spot-instances \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

### Auto-termination

```bash
# Set auto-termination after 2 hours
echo "sudo shutdown -h +120" | at now

# Or in user data
#!/bin/bash
cd /home/ubuntu/cs627
make tokens
make train
aws s3 sync OptimalWeights/ s3://your-bucket/weights/
sudo shutdown -h now
```

## Monitoring

### CloudWatch Metrics

```bash
# Enable detailed monitoring
aws ec2 monitor-instances --instance-ids $INSTANCE_ID

# View GPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name GPUUtilization \
  --dimensions Name=InstanceId,Value=$INSTANCE_ID \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T01:00:00Z \
  --period 300 \
  --statistics Maximum
```

### Instance Logs

```bash
# View training progress
tail -f Results/adapter_training/*.log

# Check GPU usage
nvidia-smi -l 1

# Monitor system
htop
```

## Cleanup

```bash
# Terminate instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Delete security group
aws ec2 delete-security-group --group-name cs627-training

# Clean up S3 (careful!)
aws s3 rm s3://your-bucket/temp/ --recursive
```

## Troubleshooting

### "No CUDA devices"
```bash
# Check GPU
nvidia-smi

# Reinstall CUDA drivers
sudo apt install nvidia-driver-525
sudo reboot
```

### "Out of memory"
```bash
# Reduce batch size
BATCH_SIZE=16 make tokens
BATCH_SIZE=128 make train
```

### "Connection timeout"
```bash
# Check security group
aws ec2 describe-security-groups --group-names cs627-training

# Check instance status
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID
```

## Automated Setup Scripts

### Quick Setup with Scripts

```bash
# 1. Setup all AWS resources automatically
./scripts/setup_aws_resources.sh --region us-east-1

# 2. Source the generated configuration
source .env.aws

# 3. Launch a training instance
./scripts/cloud/launch_training_instance.sh \
  --instance-type g5.4xlarge \
  --key-name $KEY_NAME \
  --security-group $SECURITY_GROUP_ID

# 4. Upload tokens to S3
aws s3 cp data/pregenerated_tokens/ s3://$S3_BUCKET/tokens/ --recursive

# 5. After training, cleanup resources
./scripts/cleanup_aws_resources.sh
```

### Configuration Files

- **`.env.aws.example`**: Template for AWS configuration
- **`.env.aws`**: Your local AWS configuration (created by setup script)

## Cost-Optimized Configurations

| Workload | Instance | Storage | Token Strategy | Est. Cost |
|----------|----------|---------|----------------|-----------|
| **Development** | t3.xlarge (CPU) | Local | Generate locally | $0.17/hr |
| **Single Run** | g5.4xlarge | S3 Standard | Generate on cloud | ~$2 total |
| **Multiple Runs** | g5.4xlarge | S3 Standard | Pre-generate & reuse | $0.50/run |
| **Team Training** | g5.4xlarge | S3 Standard IA | Shared tokens | $0.50/run + $0.02/mo |
| **Production** | g5.8xlarge | S3 + CloudFront | Cached globally | $1/run + CDN costs |

## Best Practices

### Security
1. Never commit `.env.aws` to version control
2. Use IAM roles instead of access keys
3. Restrict S3 bucket access with policies
4. Enable MFA for production AWS accounts

### Cost Management
1. Always terminate instances when done
2. Use lifecycle policies for S3 data
3. Consider spot instances for non-critical training
4. Set up billing alerts

### Performance
1. Pre-generate tokens locally for multiple experiments
2. Use S3 Transfer Acceleration for large uploads
3. Enable S3 multipart uploads for files >100MB
4. Compress tokens before uploading (30-50% size reduction)

## Troubleshooting

### Common Issues

**"Access Denied" errors**
- Ensure IAM role is attached to instance
- Check S3 bucket policies
- Verify security group allows necessary ports

**"No space left on device"**
- Increase EBS volume size
- Clear Docker cache: `docker system prune -a`
- Remove old logs: `rm -rf Results/logs/*`

**Token upload/download slow**
- Use S3 Transfer Acceleration
- Compress tokens: `tar -czf tokens.tar.gz data/pregenerated_tokens/`
- Use parallel uploads: `aws s3 cp --recursive --exclude "*" --include "*.h5"`

## Next Steps

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training workflows
- [DOCKER.md](DOCKER.md) - Container usage
- [SETUP.md](SETUP.md) - Local setup
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Extended troubleshooting