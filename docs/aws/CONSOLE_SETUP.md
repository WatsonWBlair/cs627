# AWS Console Setup (Web UI)

Step-by-step AWS Management Console instructions for users who prefer the web interface.

## Overview

```
1. IAM Role ──► 2. S3 Bucket ──► 3. Security Group ──► 4. EC2 Instance ──► 5. Training
```

## Step 1: IAM Role Configuration

### 1.1 Navigate to IAM
1. Open [AWS Console](https://console.aws.amazon.com)
2. Search for "IAM" in the search bar
3. Click **IAM**

### 1.2 Create Role
1. Click **Roles** in left sidebar
2. Click **Create role**
3. Select **AWS service** → **EC2** → **Next**

### 1.3 Add Permissions
1. Search and select:
   - **AmazonS3FullAccess**
   - **CloudWatchAgentServerPolicy**
2. Click **Next**
3. Role name: `cs627-ec2-role`
4. Click **Create role**

### 1.4 Note the Role ARN
Click the role and copy the ARN: `arn:aws:iam::ACCOUNT-ID:role/cs627-ec2-role`

## Step 2: S3 Bucket Setup

### 2.1 Navigate to S3
1. Search for "S3" → Click **S3**

### 2.2 Create Bucket
1. Click **Create bucket**
2. Configuration:
   - **Bucket name**: `cs627-training-[yourname]-[date]`
   - **Region**: us-east-1
   - **Block Public Access**: Keep all checked
   - **Versioning**: Enable
3. Click **Create bucket**

### 2.3 Create Folders
In your bucket, create these folders:
- `tokens/`
- `weights/`
- `results/`
- `logs/`

### 2.4 Lifecycle Policy (Optional)
1. Go to **Management** tab
2. Click **Create lifecycle rule**
3. Name: `TransitionOldTokens`
4. Prefix: `tokens/`
5. Transitions: 30 days → Standard-IA, 90 days → Glacier

## Step 3: Security Group

### 3.1 Navigate to EC2
1. Search for "EC2" → Click **EC2**

### 3.2 Create Security Group
1. Click **Security Groups** → **Create security group**
2. Name: `cs627-training`
3. VPC: Default VPC

### 3.3 Inbound Rules
Add these rules:

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | My IP | SSH access |
| Custom TCP | 8888 | My IP | Jupyter |
| Custom TCP | 6006 | My IP | TensorBoard |

4. Click **Create security group**

## Step 4: ECR Repository (Optional)

### 4.1 Navigate to ECR
1. Search for "ECR" → Click **Elastic Container Registry**

### 4.2 Create Repository
1. Click **Create repository**
2. Name: `cs627-svs`
3. Click **Create repository**

### 4.3 Push Commands
Click **View push commands** and save for later.

## Step 5: Launch EC2 Instance

### 5.1 Launch Instance
1. In EC2, click **Instances** → **Launch instances**

### 5.2 Configure Instance

**Name**: `cs627-training-instance`

**AMI**:
1. Click **Browse more AMIs**
2. Search "Deep Learning AMI GPU PyTorch"
3. Select latest version

**Instance Type**: `g5.4xlarge`
- 16 vCPUs, 64GB RAM, A10G GPU (24GB)
- Cost: ~$1.62/hour

**Key Pair**:
1. Create new or select existing
2. Download .pem file

**Network Settings**:
1. Click **Edit**
2. Auto-assign public IP: **Enable**
3. Security group: Select `cs627-training`

**Storage**: 100 GB gp3

**Advanced Details**:
1. IAM instance profile: `cs627-ec2-role`
2. User data (optional):
```bash
#!/bin/bash
apt-get update && apt-get install -y git htop
git clone https://github.com/WatsonWBlair/cs627.git /home/ubuntu/cs627
chown -R ubuntu:ubuntu /home/ubuntu/cs627
```

### 5.3 Launch
Click **Launch instance** and wait 2-3 minutes.

## Step 6: Connect and Train

### 6.1 Connect via SSH
1. Select instance → Click **Connect**
2. Follow SSH client instructions:
```bash
chmod 400 cs627-training-key.pem
ssh -i cs627-training-key.pem ubuntu@[PUBLIC-IP]
```

### 6.2 Configure AWS CLI
```bash
aws configure set region us-east-1
aws s3 ls s3://cs627-training-[your-bucket]/  # Test access
```

### 6.3 Run Training
```bash
cd ~/cs627
python src/Training/pregenerate_tokens.py  # ~30 min
python src/Training/train_adapters.py --mode encoder  # ~10 min
python src/Training/train_adapters.py --mode decoder  # ~10 min

# Upload results
aws s3 cp OptimalWeights/ s3://cs627-training-[bucket]/weights/ --recursive
```

## Step 7: Cleanup

### 7.1 Terminate Instance
1. Select instance → **Instance state** → **Terminate**

### 7.2 Delete Resources (Optional)
- **S3**: Empty bucket → Delete bucket
- **Security Group**: Select → Actions → Delete
- **IAM Role**: Roles → Select → Delete
- **ECR**: Select repository → Delete

## Tips

1. **Bookmark pages**: EC2, S3, CloudWatch
2. **Use tags**: Always tag with `Project=CS627`
3. **Set billing alerts**: Billing → Budgets
4. **Save as template**: EC2 → Launch Templates → Create from instance

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common AWS issues.
