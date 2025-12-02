# Cloud Deployment Guide

This guide explains how to deploy and train the Semantic-Vector Space encoder alignment project on cloud GPU instances.

## Quick Start

### 1. Launch Cloud Instance

**Recommended Specs:**
- **GPU**: NVIDIA T4, V100, A10, or A100
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ SSD
- **OS**: Ubuntu 20.04 or 22.04 LTS

**Popular Cloud Providers:**
- [AWS EC2](https://aws.amazon.com/ec2/) - `g4dn.xlarge` or `p3.2xlarge`
- [Google Cloud Platform](https://cloud.google.com/compute) - `n1-standard-4` with T4 GPU
- [Azure](https://azure.microsoft.com/en-us/products/virtual-machines/) - `NC6s_v3`
- [Lambda Labs](https://lambdalabs.com/) - GPU Cloud instances (cost-effective)
- [Paperspace](https://www.paperspace.com/) - GPU instances with Jupyter

### 2. Clone Repository

```bash
# SSH into your instance
ssh user@your-instance-ip

# Clone the repository
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
```

### 3. Run Setup Script

```bash
# Make script executable
chmod +x setup_and_train.sh

# Run with default settings
./setup_and_train.sh
```

That's it! The script will:
1. ✓ Install all dependencies (Python, libraries, CUDA tools)
2. ✓ Download CMU-MultimodalSDK
3. ✓ Download and preprocess CMU-MOSI dataset (~2GB)
4. ✓ Train encoders with MoCo contrastive learning
5. ✓ Save adapter weights and training logs

### 4. Monitor Training

Training progress will be displayed in real-time:

```
Epoch 1/100:  Loss: 2.45
Epoch 2/100:  Loss: 2.21
...
Epoch 50/100: Loss: 0.87
```

Training logs are saved to `results/encoder_alignment/training_YYYYMMDD_HHMMSS.log`

## Advanced Usage

### Custom Training Configuration

```bash
# Train with larger batch size and queue (requires more GPU memory)
./setup_and_train.sh --batch-size 64 --queue-size 16384 --epochs 200

# Train for fewer epochs (testing)
./setup_and_train.sh --epochs 10

# Skip dependency installation (if already set up)
./setup_and_train.sh --skip-setup

# Skip data download (if already downloaded)
./setup_and_train.sh --skip-data

# Setup only (don't train)
./setup_and_train.sh --skip-train
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size SIZE` | Training batch size | 32 |
| `--queue-size SIZE` | Memory queue size | 4096 |
| `--epochs NUM` | Number of training epochs | 100 |
| `--skip-setup` | Skip dependency installation | false |
| `--skip-data` | Skip data download/preprocessing | false |
| `--skip-train` | Skip training (setup only) | false |
| `--data-path PATH` | Data directory | data/cmumosi |
| `--output-dir DIR` | Output directory | results/encoder_alignment |
| `--help` | Show help message | - |

### Batch Size Recommendations by GPU

| GPU | Memory | Recommended Batch Size | Queue Size |
|-----|--------|------------------------|------------|
| T4 | 16GB | 16-32 | 2048-4096 |
| V100 | 16GB | 32-64 | 4096-8192 |
| V100 | 32GB | 64-128 | 8192-16384 |
| A10 | 24GB | 64-96 | 8192-16384 |
| A100 | 40GB | 128-256 | 16384-65536 |

## Step-by-Step Manual Setup

If you prefer manual setup or need to customize the process:

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    build-essential \
    libsndfile1 \
    ffmpeg
```

**RHEL/CentOS:**
```bash
sudo yum install -y \
    python3 \
    python3-pip \
    git \
    wget \
    gcc \
    gcc-c++ \
    make \
    libsndfile \
    ffmpeg
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install CMU-MultimodalSDK

```bash
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK
pip install .
cd ..
```

### 4. Download Dataset

```bash
python3 -c "
from src.Training.Data_Wrangling.mosi_dataset import download_mosi, preprocess_mosi
download_mosi('data/cmumosi')
preprocess_mosi('data/cmumosi')
"
```

### 5. Run Training

```bash
python3 src/Training/train_encoder_alignment.py
```

## Monitoring and Logging

### View GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### View Training Logs

```bash
# Follow training log in real-time
tail -f results/encoder_alignment/training_*.log

# View last 100 lines
tail -n 100 results/encoder_alignment/training_*.log
```

### Check Training Progress

```bash
# List checkpoints
ls -lh results/encoder_alignment/checkpoint-*/

# List adapter weights
ls -lh AdapterWeights/
```

## Downloading Results

### Download Adapter Weights

**Using SCP:**
```bash
# From your local machine
scp -r user@instance-ip:~/cs627/AdapterWeights/ ./
```

**Using rsync:**
```bash
# From your local machine
rsync -avz user@instance-ip:~/cs627/AdapterWeights/ ./AdapterWeights/
```

### Download Training Logs

```bash
# From your local machine
scp user@instance-ip:~/cs627/results/encoder_alignment/*.log ./
```

## Cost Optimization

### 1. Use Spot/Preemptible Instances

- **AWS**: Spot Instances (up to 90% savings)
- **GCP**: Preemptible VMs (up to 80% savings)
- **Azure**: Spot VMs (up to 90% savings)

**Important**: Enable checkpointing to resume training if instance is interrupted.

### 2. Start Small, Scale Up

```bash
# Test with small configuration first
./setup_and_train.sh --epochs 1 --batch-size 16

# Once working, scale up
./setup_and_train.sh --epochs 100 --batch-size 64
```

### 3. Shutdown When Done

```bash
# After training completes, download results and terminate instance
# AWS
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# GCP
gcloud compute instances delete instance-name --zone=us-central1-a
```

## Troubleshooting

### Out of Memory (OOM)

**Error**: `CUDA out of memory`

**Solution**: Reduce batch size or queue size
```bash
./setup_and_train.sh --batch-size 16 --queue-size 2048
```

### Slow Data Download

**Issue**: CMU-MOSI download is slow or fails

**Solution**: Download dataset manually and place in `data/cmumosi/`

### CUDA Not Available

**Issue**: Training uses CPU instead of GPU

**Check**:
```bash
# Verify NVIDIA drivers
nvidia-smi

# Verify PyTorch sees GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Solution**: Install CUDA toolkit and compatible PyTorch version
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'mmsdk'`

**Solution**: Install CMU-MultimodalSDK
```bash
cd CMU-MultimodalSDK && pip install . && cd ..
```

## Docker Deployment (Alternative)

Coming soon: Docker container for reproducible deployments.

## Estimated Costs

**Training Time**: ~4-8 hours (depends on GPU and configuration)

**Example Costs**:
- **AWS g4dn.xlarge** (T4): ~$0.50/hour → $2-4 per training run
- **AWS p3.2xlarge** (V100): ~$3.00/hour → $12-24 per training run
- **Lambda Labs** (A10): ~$0.60/hour → $2.40-4.80 per training run

**Data Transfer**: Minimal (dataset is ~2GB, results are <1GB)

## Next Steps

After training completes:

1. **Download adapter weights** from `AdapterWeights/`
2. **Review training logs** in `results/encoder_alignment/`
3. **Evaluate cross-modal retrieval** metrics
4. **Use trained encoders** for downstream tasks

See [Training README](src/Training/README.md) for detailed evaluation metrics and next steps.

## Support

For issues or questions:
- Check [CLAUDE.md](CLAUDE.md) for project documentation
- Review [Training README](src/Training/README.md) for training details
- See [Training QUICKSTART](src/Training/QUICKSTART.md) for quick start guide
