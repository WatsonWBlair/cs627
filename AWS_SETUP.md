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
