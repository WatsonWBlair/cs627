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
3. ✓ Download and preprocess CMU-MOSI dataset (~10GB raw videos)
4. ✓ Train encoders with MoCo contrastive learning
5. ✓ Save adapter weights and training logs

**⚠️ Note**: This downloads 10GB on your GPU instance. See "Recommended: Preprocess Locally" below for a more cost-effective approach (only transfer 500MB).

### 4. Monitor Training

Training progress will be displayed in real-time:

```
Epoch 1/100:  Loss: 2.45
Epoch 2/100:  Loss: 2.21
...
Epoch 50/100: Loss: 0.87
```

Training logs are saved to `results/encoder_alignment/training_YYYYMMDD_HHMMSS.log`

## Recommended: Preprocess Locally, Transfer to Cloud

**Why?** Downloading 10GB of videos on an expensive GPU instance is wasteful. Instead:
- Download and extract data on your local machine (free, can run overnight)
- Transfer only ~500MB of segmented data to cloud (20x less data transfer)
- Start training immediately on GPU (no wasted GPU hours)

### Step 1: Preprocess on Local Machine

```bash
# On your local machine
cd cs627

# Download MOSI metadata
python -c "
from src.Training.Data_Wrangling.mosi_dataset import download_mosi
download_mosi('data/cmumosi/mosi/')
"

# Download raw videos (~10GB, can run overnight)
python scripts/data_wrangling/download_all_mosi_videos.py

# Extract segments (~500MB total)
python scripts/data_wrangling/extract_all_segments.py
```

### Step 2: Transfer Data to Cloud Instance

```bash
# From your local machine, transfer only the extracted segments
rsync -avz --progress \
  data/cmumosi/ \
  user@instance-ip:~/cs627/data/cmumosi/
```

**Transfer size**: Only ~500MB (vs 10GB for raw videos)
**Transfer time**: ~5-10 minutes on typical broadband (vs 1-2 hours for raw videos)

### Step 3: Train on Cloud

```bash
# SSH into cloud instance
ssh user@instance-ip

# Install dependencies only (skip data download)
cd cs627
./setup_and_train.sh --skip-data

# Or manually:
pip install -r requirements.txt
python3 src/Training/train_encoders.py
```

**Cost Savings**: Save 1-2 hours of GPU time ($2-6 on most instances)

## Automated Transfer Scripts

For easier cloud management, we provide automated scripts to transfer data and weights:

### Quick Setup

```bash
# 1. Configure cloud credentials (one time)
cp .env.example .env
nano .env  # Fill in CLOUD_USER, CLOUD_HOST, CLOUD_SSH_KEY

# 2. Upload preprocessed data to cloud
./scripts/cloud/upload_to_cloud.sh

# 3. SSH and train
ssh $CLOUD_USER@$CLOUD_HOST
cd ~/cs627
python src/Training/train_encoders.py

# 4. Download trained weights (from local machine)
./scripts/cloud/download_from_cloud.sh
```

### Upload Script (`scripts/cloud/upload_to_cloud.sh`)

Uploads preprocessed data (~500MB) to cloud instance:

```bash
# Upload data and weights (default)
./scripts/cloud/upload_to_cloud.sh

# Upload only data (first time training)
./scripts/cloud/upload_to_cloud.sh --data-only

# Preview transfer without executing
./scripts/cloud/upload_to_cloud.sh --dry-run
```

### Download Script (`scripts/cloud/download_from_cloud.sh`)

Downloads trained weights and results from cloud:

```bash
# Download everything (weights + logs + metrics)
./scripts/cloud/download_from_cloud.sh

# Download only trained weights
./scripts/cloud/download_from_cloud.sh --weights-only

# Download only training logs/metrics
./scripts/cloud/download_from_cloud.sh --results-only
```

**See [scripts/cloud/README.md](scripts/cloud/README.md) for complete documentation.**

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

### 4. Download and Prepare Dataset

```bash
# Download MOSI metadata (text transcripts, labels)
python3 -c "
from src.Training.Data_Wrangling.mosi_dataset import download_mosi
download_mosi('data/cmumosi/mosi/')
"

# Download raw videos (~10GB)
python3 scripts/data_wrangling/download_all_mosi_videos.py

# Extract audio segments and video frames (~500MB total)
python3 scripts/data_wrangling/extract_all_segments.py
```

**Note**: This downloads ~10GB of raw video, then extracts ~500MB of segmented audio/frames for training.

### 5. Run Training

```bash
python3 src/Training/train_encoders.py
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
ls -lh OptimalWeights/
ls -lh CandidateWeights/
```

## Weight Management Workflow

This project uses a two-tier weight management system to enable safe experimentation and version control:

### Weight Directories

1. **OptimalWeights/** - Production-ready adapter weights
   - Verified to meet performance criteria
   - Used by default when loading encoders/decoders
   - Version controlled in Git (tracked in repository)
   - Example: `OptimalWeights/facebook_bart-base_text_enc_weights.pth`

2. **CandidateWeights/** - Experimental weights from training runs
   - Organized by instance and timestamp: `CandidateWeights/{instance_id}_{timestamp}/`
   - NOT version controlled (in `.gitignore`)
   - Requires evaluation before promotion to OptimalWeights
   - Example: `CandidateWeights/aws-p3-2xlarge_20240115_143022/facebook_bart-base_text_enc_weights.pth`

### Training Workflow

**During Cloud Training:**

1. Training script automatically saves to `CandidateWeights/{instance}_{timestamp}/`
2. Each training run creates a new timestamped directory
3. Original OptimalWeights remain unchanged

```bash
# Training saves here:
CandidateWeights/
└── aws-gpu-instance_20240115_143022/
    ├── facebook_bart-base_text_enc_weights.pth
    ├── openai_whisper-small_audio_enc_weights.pth
    └── nlpconnect_vit-gpt2-image-captioning_image_enc_weights.pth
```

**After Training:**

1. Download candidate weights from cloud
2. Run evaluation to compare against current OptimalWeights
3. If performance improves, promote candidates to OptimalWeights
4. Commit promoted weights to repository

```bash
# 1. Download candidate weights
scp -r user@instance-ip:~/cs627/CandidateWeights/ ./

# 2. Evaluate candidate weights
python scripts/evaluate_weights.py \
  --candidate-dir CandidateWeights/aws-gpu-instance_20240115_143022/ \
  --optimal-dir OptimalWeights/

# 3. Promote if better (overwrites OptimalWeights)
python scripts/promote_weights.py \
  --source CandidateWeights/aws-gpu-instance_20240115_143022/ \
  --dest OptimalWeights/

# 4. Commit to repository
git add OptimalWeights/
git commit -m "Promote weights from training run 20240115_143022"
```

### Best Practices

**DO:**
- ✅ Always evaluate candidate weights before promotion
- ✅ Keep training runs organized with meaningful instance IDs
- ✅ Document evaluation metrics when promoting weights
- ✅ Commit OptimalWeights to Git after promotion
- ✅ Archive old OptimalWeights before overwriting (optional)

**DON'T:**
- ❌ Never commit CandidateWeights to Git (too large, experimental)
- ❌ Don't overwrite OptimalWeights without evaluation
- ❌ Don't delete CandidateWeights until confirmed worse than OptimalWeights
- ❌ Don't skip evaluation metrics in commit messages

### Rollback Strategy

If new weights perform worse than expected:

```bash
# Revert to previous OptimalWeights
git log --oneline -- OptimalWeights/  # Find previous commit
git checkout <commit-hash> -- OptimalWeights/
git commit -m "Rollback to previous optimal weights"
```

## Downloading Results

### Automated Download (Recommended)

Use the automated download script:

```bash
# Download everything (weights + logs + metrics)
./scripts/cloud/download_from_cloud.sh

# Download only weights
./scripts/cloud/download_from_cloud.sh --weights-only
```

See [Automated Transfer Scripts](#automated-transfer-scripts) section above for details.

### Manual Download (Alternative)

**Using SCP:**
```bash
# From your local machine
scp -r user@instance-ip:~/cs627/OptimalWeights/ ./
scp -r user@instance-ip:~/cs627/results/ ./
```

**Using rsync:**
```bash
# From your local machine
rsync -avz user@instance-ip:~/cs627/OptimalWeights/ ./OptimalWeights/
rsync -avz user@instance-ip:~/cs627/results/ ./results/
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

**Issue**: CMU-MOSI download is slow or fails on cloud instance

**Solution**: Use the recommended approach - preprocess locally and transfer segmented data
- See "Recommended: Preprocess Locally, Transfer to Cloud" section above
- Only transfers 500MB instead of 10GB (20x faster)

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

**Example Costs (GPU time only)**:
- **AWS g4dn.xlarge** (T4): ~$0.50/hour → $2-4 per training run
- **AWS p3.2xlarge** (V100): ~$3.00/hour → $12-24 per training run
- **Lambda Labs** (A10): ~$0.60/hour → $2.40-4.80 per training run

**Data Transfer Costs**:
- **If downloading on cloud**: 10GB download + 1-2 hours extra GPU time = $0.50-$6 extra
- **If preprocessing locally**: 500MB transfer only = minimal cost

**💡 Cost Optimization**: Preprocess locally to save 1-2 hours of GPU time ($0.50-$6 per run)

## Next Steps

After training completes:

1. **Download adapter weights** from `OptimalWeights/`
2. **Review training logs** in `results/encoder_alignment/`
3. **Evaluate cross-modal retrieval** metrics
4. **Use trained encoders** for downstream tasks

See [Training README](src/Training/README.md) for detailed evaluation metrics and next steps.

## Support

For issues or questions:
- Check [CLAUDE.md](CLAUDE.md) for project documentation
- Review [Training README](src/Training/README.md) for training details
- See [Training QUICKSTART](src/Training/QUICKSTART.md) for quick start guide
