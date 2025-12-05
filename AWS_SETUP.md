# AWS Training Setup

Configuration for training on AWS EC2 GPU instances.

## Recommended Configuration

| Component | Recommendation |
|-----------|---------------|
| **Instance** | g5.4xlarge |
| **GPU** | NVIDIA A10G (24GB VRAM) |
| **AMI** | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Amazon Linux 2023) |
| **Storage** | 200GB gp3 |

## Instance Type Comparison

| Instance | GPU | VRAM | vCPUs | RAM | Cost/hr |
|----------|-----|------|-------|-----|---------|
| **g5.4xlarge** | A10G | 24GB | 16 | 64GB | ~$1.62 |
| g5.2xlarge | A10G | 24GB | 8 | 32GB | ~$1.21 |
| g4dn.xlarge | T4 | 16GB | 4 | 16GB | ~$0.53 |

## Quick Start

```bash
# 1. Launch instance via AWS Console
#    - AMI: Search "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8"
#    - Instance type: g5.4xlarge
#    - Storage: 200GB gp3

# 2. SSH to instance
ssh ec2-user@<instance-ip>

# 3. Clone and setup
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
pip install -r requirements.txt

# 4. Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..

# 5. Start training
python src/Training/train_encoders.py
```

## Monitoring

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training logs
tail -f training.log
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size: `BATCH_SIZE=16` environment variable

### Disk Space
Ensure 200GB+ storage. Models require ~3GB, training data ~60GB.

## Cost Optimization

- Use spot instances for up to 70% savings
- Set up auto-shutdown to prevent runaway costs
- g5.2xlarge is a good budget alternative

## Training Performance

| Task | Time on A10G |
|------|--------------|
| Single Epoch | 5-7 min |
| Full Training (30 epochs) | 3-4 hours |

---

## S3 Data Staging (Recommended)

Using S3 for training data eliminates slow rsync transfers from your local machine.

### One-Time Setup

#### 1. Create S3 Bucket

```bash
# Via AWS CLI
aws s3 mb s3://cs627-svs-data --region us-east-1

# Enable versioning (for weight history/rollback)
aws s3api put-bucket-versioning \
  --bucket cs627-svs-data \
  --versioning-configuration Status=Enabled
```

#### 2. Upload Training Data

```bash
# Upload training data (~3.3GB) - run once from your local machine
aws s3 sync data/cmumosi/ s3://cs627-svs-data/data/cmumosi/ --exclude "videos/*"

# Upload current OptimalWeights
aws s3 sync OptimalWeights/ s3://cs627-svs-data/OptimalWeights/
```

#### 3. Create IAM Role for EC2

Create an IAM role named `cs627-training-role` with this policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::cs627-svs-data",
        "arn:aws:s3:::cs627-svs-data/*"
      ]
    }
  ]
}
```

#### 4. Launch EC2 with IAM Role

When launching via AWS Console:
1. Choose Deep Learning AMI
2. In "Advanced details" → IAM instance profile → select `cs627-training-role`
3. No AWS credentials needed in code (uses instance metadata)

### Training with S3

```bash
# On EC2 instance
docker pull watsonblair/cs627-svs:latest

# Run training (data auto-downloads from S3)
docker run --gpus all \
  -e USE_S3=1 \
  -e S3_BUCKET=cs627-svs-data \
  -v $(pwd)/CandidateWeights:/workspace/CandidateWeights \
  -v $(pwd)/training_reports:/workspace/training_reports \
  watsonblair/cs627-svs:latest \
  python src/Training/train_encoders.py
```

### Promoting Weights to S3

After training, evaluate and promote weights:

```bash
# Review training results
cat training_reports/training_metrics.json

# Promote best weights to S3 OptimalWeights
python scripts/cloud/promote_weights_s3.py \
  --best \
  --bucket cs627-svs-data
```

### Sync Weights to Local Machine

```bash
# Download from S3
python scripts/cloud/promote_weights_s3.py --download --bucket cs627-svs-data

# Or via AWS CLI
aws s3 sync s3://cs627-svs-data/OptimalWeights/ OptimalWeights/
```

### S3 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_S3` | `0` | Set to `1` to enable S3 data loading |
| `S3_BUCKET` | `cs627-svs-data` | S3 bucket name |
| `S3_REGION` | `us-east-1` | AWS region |
| `S3_DATA_PREFIX` | `data/cmumosi/` | S3 prefix for training data |
| `S3_WEIGHTS_PREFIX` | `OptimalWeights/` | S3 prefix for weights |

### Time Comparison

| Method | Data Transfer Time |
|--------|-------------------|
| rsync from local | 35+ min (depends on connection) |
| S3 on EC2 | 1-2 min (10+ Gbps internal) |
