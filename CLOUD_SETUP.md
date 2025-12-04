# Cloud Training Setup Guide

This guide provides optimal configurations for training the Semantic-Vector Space models on cloud GPU instances.

## Table of Contents
- [Recommended Configuration](#recommended-configuration)
- [AWS EC2 Setup](#aws-ec2-setup)
- [Google Cloud Platform](#google-cloud-platform)
- [Microsoft Azure](#microsoft-azure)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

## Recommended Configuration

After extensive testing, we recommend the following configuration for optimal performance and ease of setup:

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **GPU** | NVIDIA A10G (24GB) | Best price/performance for transformer training |
| **Instance** | g5.4xlarge (AWS) | 16 vCPUs, 64GB RAM, NVMe SSD |
| **AMI** | Deep Learning AMI | Pre-installed CUDA, PyTorch, dependencies |
| **Storage** | 200GB gp3 | Sufficient for datasets + checkpoints |
| **OS** | Ubuntu 22.04 | Better compatibility than 24.04 |

### Training Performance Benchmarks

| Task | Time on A10G | Time on CPU | Speedup |
|------|--------------|-------------|---------|
| Single Encoder Epoch | 5-7 minutes | 45-60 minutes | ~9x |
| Full Training (30 epochs) | 3-4 hours | 24-30 hours | ~8x |
| Batch Processing | 200ms/batch | 1800ms/batch | ~9x |

## AWS EC2 Setup

### Quick Start (Recommended)

Use the AWS Deep Learning AMI for zero-configuration setup:

```bash
# Launch instance with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-04f3e35dc85e9423b \
  --instance-type g5.4xlarge \
  --key-name your-key-name \
  --security-group-ids sg-XXXXXXXXX \
  --subnet-id subnet-XXXXXXXXX \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cs627-training}]' \
  --output json | jq -r '.Instances[0].PublicIpAddress'
```

### Alternative AMI Options

1. **AWS Deep Learning Base GPU AMI (Ubuntu 22.04)**
   - AMI ID: `ami-0c02fb55731490381` (us-east-1)
   - Pros: Minimal, clean environment
   - Cons: Requires pip installations

2. **NVIDIA GPU-Optimized AMI**
   - Search for: "NVIDIA GPU-Optimized"
   - Pros: Latest NVIDIA drivers
   - Cons: May lack Python ML packages

3. **Standard Ubuntu AMI** (Not Recommended)
   - Requires manual NVIDIA driver installation
   - Needs reboot after driver setup
   - Multiple dependency issues encountered

### Instance Type Comparison

| Instance | GPU | VRAM | vCPUs | RAM | Cost/hr | Notes |
|----------|-----|------|-------|-----|---------|-------|
| **g5.4xlarge** | A10G | 24GB | 16 | 64GB | $1.62 | **Best overall** |
| g5.2xlarge | A10G | 24GB | 8 | 32GB | $1.21 | Good budget option |
| g5.8xlarge | A10G | 24GB | 32 | 128GB | $2.44 | For large batches |
| g4dn.xlarge | T4 | 16GB | 4 | 16GB | $0.53 | Budget training |
| g4ad.4xlarge | Radeon Pro V520 | 8GB | 16 | 64GB | $0.87 | AMD (limited support) |
| p3.2xlarge | V100 | 16GB | 8 | 61GB | $3.06 | Older but powerful |

### Security Group Configuration

Create a security group with these rules:

```bash
aws ec2 create-security-group \
  --group-name cs627-training \
  --description "Security group for CS627 training instances"

# Add SSH access
aws ec2 authorize-security-group-ingress \
  --group-name cs627-training \
  --protocol tcp \
  --port 22 \
  --cidr YOUR_IP/32

# Add Jupyter (if needed)
aws ec2 authorize-security-group-ingress \
  --group-name cs627-training \
  --protocol tcp \
  --port 8888 \
  --cidr YOUR_IP/32
```

### Deployment Script

Once your instance is running, use our setup script:

```bash
# Make script executable
chmod +x scripts/setup_remote_instance.sh

# Deploy and configure
./scripts/setup_remote_instance.sh <instance-ip>

# SSH to instance
ssh ubuntu@<instance-ip>

# Start training
cd ~/cs627
source venv/bin/activate  # If using standard AMI
python src/Training/train_encoders.py
```

## Google Cloud Platform

### Recommended Configuration

```bash
# Create instance with Deep Learning VM
gcloud compute instances create cs627-training \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd
```

### Instance Types

| Machine Type | GPU | VRAM | Cost/hr | Notes |
|--------------|-----|------|---------|-------|
| a2-highgpu-1g | A100 | 40GB | $2.87 | Overkill for our needs |
| n1-standard-4 + T4 | T4 | 16GB | $0.35 | Good budget option |
| n1-standard-8 + V100 | V100 | 16GB | $2.48 | Good performance |

## Microsoft Azure

### Recommended Configuration

```bash
# Create resource group
az group create --name cs627-training --location eastus

# Create VM with GPU
az vm create \
  --resource-group cs627-training \
  --name cs627-vm \
  --image microsoft-dsvm:ubuntu-hpc:2204:latest \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

### Instance Types

| Size | GPU | VRAM | vCPUs | RAM | Cost/hr |
|------|-----|------|-------|-----|---------|
| NC6s_v3 | V100 | 16GB | 6 | 112GB | $3.06 |
| NC4as_T4_v3 | T4 | 16GB | 4 | 28GB | $0.53 |
| NV4as_v4 | Radeon MI25 | 8GB | 4 | 14GB | $0.25 |

## Cost Optimization

### 1. Use Spot Instances (AWS)

```bash
# Request spot instance (up to 70% cheaper)
aws ec2 request-spot-instances \
  --instance-count 1 \
  --type "persistent" \
  --launch-specification file://spot-spec.json
```

### 2. Auto-shutdown Script

Add to your training script to prevent runaway costs:

```python
import time
import subprocess

# Auto-shutdown after 4 hours
def auto_shutdown(hours=4):
    time.sleep(hours * 3600)
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])

# Start in background
import threading
shutdown_thread = threading.Thread(target=auto_shutdown, args=(4,))
shutdown_thread.daemon = True
shutdown_thread.start()
```

### 3. Use Preemptible Instances (GCP)

```bash
# Add --preemptible flag (up to 80% cheaper)
gcloud compute instances create ... --preemptible
```

### 4. Storage Optimization

- Use gp3 instead of gp2 (AWS): 20% cheaper
- Delete unneeded snapshots
- Use S3/GCS for long-term checkpoint storage

## Monitoring Training

### GPU Utilization

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed metrics
nvidia-smi dmon -s pucvmet -i 0
```

### Training Progress

```bash
# Monitor training logs
tail -f ~/cs627/training.log

# Check loss curves
grep "Loss:" ~/cs627/training.log | tail -20
```

### Resource Usage

```bash
# System resources
htop

# Disk usage
df -h

# Network (if using distributed training)
iftop
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

```python
# Reduce batch size in training config
BATCH_SIZE = 16  # Instead of 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

#### 2. Slow Data Loading

```python
# Increase DataLoader workers
DataLoader(..., num_workers=4, pin_memory=True)

# Pre-cache datasets to SSD
cache_dir = "/tmp/cache"
dataset.save_to_disk(cache_dir)
```

#### 3. Driver Issues (Vanilla AMI)

```bash
# If nvidia-smi fails
sudo apt update
sudo apt install -y nvidia-driver-550
sudo reboot

# After reboot
nvidia-smi  # Should now work
```

#### 4. Dependency Conflicts

Use Deep Learning AMI to avoid these issues, or:

```bash
# Create fresh conda environment
conda create -n cs627 python=3.10
conda activate cs627
pip install -r requirements.txt
```

## Best Practices

1. **Always use Deep Learning AMIs** - Saves hours of setup time
2. **Monitor GPU utilization** - Should be >80% for efficient training
3. **Use mixed precision training** - 2x speedup with minimal accuracy loss
4. **Save checkpoints frequently** - Every 5 epochs minimum
5. **Set up auto-shutdown** - Prevent runaway costs
6. **Use spot/preemptible instances** - For non-critical training runs
7. **Test locally first** - Debug on CPU before paying for GPU time

## Performance Tips

### Optimal Hyperparameters for A10G

```python
# Tested configurations
BATCH_SIZE = 32  # Max that fits in 24GB VRAM
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION = 2  # Effective batch = 64
NUM_WORKERS = 4
MIXED_PRECISION = True
```

### Multi-GPU Training (g5.12xlarge with 4x A10G)

```python
# Use DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Launch with
# torchrun --nproc_per_node=4 train_encoders.py
```

## Support

For issues specific to cloud setup:
- AWS: Check [EC2 GPU Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html)
- GCP: See [GPU Platform Guide](https://cloud.google.com/compute/docs/gpus)
- Azure: Review [GPU VM Sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)

For project-specific issues:
- Open an issue on [GitHub](https://github.com/WatsonWBlair/cs627)
- Check existing solutions in [TROUBLESHOOTING.md](TROUBLESHOOTING.md)