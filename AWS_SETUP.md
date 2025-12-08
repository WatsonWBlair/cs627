# AWS Setup Guide

Deploy CS627 training on AWS EC2 with GPU support.

## Quick Start

### 1. Launch Instance

```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-04f3e35dc85e9423b \
  --instance-type g5.4xlarge \
  --key-name your-key \
  --security-groups default \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}"
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

### 3. Train

```bash
# Using Docker
docker pull watsonwb/cs627-svs:latest
make tokens
make train

# Or local
python src/Training/pregenerate_tokens.py
python src/Training/train_adapters.py
```

## Recommended Configuration

| Component | Specification | Cost |
|-----------|--------------|------|
| **Instance** | g5.4xlarge | $1.62/hour |
| **GPU** | NVIDIA A10G (24GB) | Included |
| **vCPUs** | 16 | Included |
| **Memory** | 64 GB | Included |
| **Storage** | 200 GB gp3 | $16/month |
| **Region** | us-east-1 | - |

**Estimated training cost**: $5-10 for full pipeline

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
- **Name**: AWS Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5.0
- **AMI ID** (us-east-1): `ami-04f3e35dc85e9423b`
- **AMI ID** (us-west-2): `ami-0c2d06d50ce30b442`

### Finding Latest AMI
```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=*Deep Learning AMI GPU PyTorch*" \
  --query 'Images[0].ImageId'
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
      "VolumeSize": 200,
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

### 6. Run Training

```bash
# Generate tokens (if needed)
make tokens

# Train adapters
make train

# Run ablation study
python cli.py ablation --configs recommended
```

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

## Next Steps

- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Training workflows
- [DOCKER.md](DOCKER.md) - Container usage
- [SETUP.md](SETUP.md) - Local setup