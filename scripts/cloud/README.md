# Cloud Transfer Scripts

Automated scripts for transferring data and trained models between your local machine and cloud GPU instances.

## Quick Start

### 1. Configure Cloud Credentials

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your cloud instance details
nano .env  # or use your preferred editor
```

Required variables in `.env`:
- `CLOUD_USER` - SSH username (e.g., ubuntu, ec2-user)
- `CLOUD_HOST` - Instance IP or hostname
- `CLOUD_SSH_KEY` - Path to SSH private key (optional if using default)
- `CLOUD_PROJECT_DIR` - Project directory on cloud instance (default: ~/cs627)

### 2. Upload Preprocessed Data to Cloud

**Recommended workflow** (saves GPU costs):

```bash
# Preprocess data locally (can run overnight, no GPU cost)
python scripts/data_wrangling/download_all_mosi_videos.py
python scripts/data_wrangling/extract_all_segments.py

# Upload only the processed segments (~500MB vs 10GB)
./scripts/cloud/upload_to_cloud.sh
```

This uploads only the segmented audio and video frames needed for training, not the raw videos.

### 3. Train on Cloud Instance

```bash
# SSH into your cloud instance
ssh $CLOUD_USER@$CLOUD_HOST

# Navigate to project
cd ~/cs627

# Install dependencies (first time only)
pip install -r requirements.txt

# Start training
python src/Training/train_encoders.py
```

### 4. Download Trained Weights

```bash
# After training completes, download results from local machine
./scripts/cloud/download_from_cloud.sh
```

This downloads:
- Trained adapter weights (`OptimalWeights/`)
- Training logs and metrics (`results/`)
- Loss curves and checkpoints (`training_reports/`)

## Script Reference

### upload_to_cloud.sh

Upload preprocessed data and optional adapter weights to cloud instance.

**Usage:**
```bash
./scripts/cloud/upload_to_cloud.sh [OPTIONS]
```

**Options:**
- `--data-only` - Upload only preprocessed data (skip weights)
- `--weights-only` - Upload only adapter weights (skip data)
- `--skip-weights` - Upload data but skip weights
- `--dry-run` - Preview what would be transferred without actually transferring
- `--help` - Show help message

**Examples:**
```bash
# Upload data and weights (default)
./scripts/cloud/upload_to_cloud.sh

# Upload only data (first time training)
./scripts/cloud/upload_to_cloud.sh --data-only

# Preview what would be transferred
./scripts/cloud/upload_to_cloud.sh --dry-run

# Resume training - upload updated weights only
./scripts/cloud/upload_to_cloud.sh --weights-only
```

**What gets uploaded:**
- `data/cmumosi/` - Preprocessed audio segments and video frames (~500MB)
- `OptimalWeights/` - Trained adapter weights (optional, for resuming training)

### download_from_cloud.sh

Download trained models, results, and logs from cloud instance to **instance-specific directory**.

**IMPORTANT**: Weights are downloaded to `CandidateWeights/{hostname}_{timestamp}/` to avoid overwriting your local development weights.

**Usage:**
```bash
./scripts/cloud/download_from_cloud.sh [OPTIONS]
```

**Options:**
- `--weights-only` - Download only adapter weights (skip results)
- `--results-only` - Download only results and logs (skip weights)
- `--dry-run` - Preview what would be transferred without actually transferring
- `--help` - Show help message

**Examples:**
```bash
# Download everything (weights + logs + metrics)
./scripts/cloud/download_from_cloud.sh

# Download only trained weights
./scripts/cloud/download_from_cloud.sh --weights-only

# Download only logs and metrics
./scripts/cloud/download_from_cloud.sh --results-only

# Preview what would be transferred
./scripts/cloud/download_from_cloud.sh --dry-run
```

**Download Structure:**
```
CandidateWeights/
└── ec2-3-80-123-45_20250104_143022/     # Instance-specific directory
    ├── OptimalWeights/                  # Trained adapter weights (.pth files)
    │   ├── text_base_weights.pth
    │   ├── audio_waveform_weights.pth
    │   ├── audio_tone_weights.pth
    │   └── image_base_weights.pth
    ├── results/                         # Training logs, eval metrics
    └── training_reports/                # Loss curves, timing analysis
        ├── training_metrics.json
        ├── loss_curves.png
        └── epoch_timing.png
```

**Promoting Weights to Production:**

After reviewing training metrics, you can promote the best weights:

```bash
# Option 1: Copy all weights to production
cp CandidateWeights/ec2-xxx_20250104_143022/OptimalWeights/* OptimalWeights/

# Option 2: Compare first, then selectively copy
python scripts/cloud/compare_weights.py \
  CandidateWeights/ec2-xxx_20250104_143022/OptimalWeights/ \
  OptimalWeights/

# Option 3: Test cloud weights before promoting
python scripts/run_evaluation.py \
  --weights-dir CandidateWeights/ec2-xxx_20250104_143022/OptimalWeights/
```

## Cost Optimization Tips

### 1. Preprocess Locally
**Save $2-6 per training run** by preprocessing data on your local machine instead of the GPU instance:

```bash
# Local machine (free)
python scripts/data_wrangling/download_all_mosi_videos.py    # ~10GB download
python scripts/data_wrangling/extract_all_segments.py        # Extract segments

# Upload only 500MB to cloud (vs 10GB)
./scripts/cloud/upload_to_cloud.sh --data-only
```

**Savings:**
- Avoids 1-2 hours of GPU time downloading/extracting
- Transfers 20x less data (500MB vs 10GB)
- Faster upload (5-10 min vs 1-2 hours)

### 2. Download Selectively
Download only what you need to minimize transfer time:

```bash
# During training - monitor progress with logs only
./scripts/cloud/download_from_cloud.sh --results-only

# After training - get final weights
./scripts/cloud/download_from_cloud.sh --weights-only
```

### 3. Use Dry Run
Preview transfers before executing to avoid mistakes:

```bash
./scripts/cloud/upload_to_cloud.sh --dry-run
./scripts/cloud/download_from_cloud.sh --dry-run
```

## Troubleshooting

### SSH Connection Failed

**Error:** `Failed to connect to <instance>`

**Solutions:**
1. Check instance is running: `aws ec2 describe-instances` or cloud provider console
2. Verify security group allows SSH (port 22)
3. Check SSH key path in `.env` is correct
4. Test SSH manually: `ssh -i ~/.ssh/key.pem user@instance-ip`

### Permission Denied

**Error:** `Permission denied (publickey)`

**Solutions:**
1. Ensure SSH key has correct permissions: `chmod 600 ~/.ssh/key.pem`
2. Verify `CLOUD_SSH_KEY` path in `.env`
3. Check username matches instance type (ubuntu, ec2-user, etc.)

### Transfer Too Slow

**Issue:** Upload/download taking too long

**Solutions:**
1. Use `--data-only` to skip large transfers
2. Compress before transfer: `tar -czf data.tar.gz data/`
3. Consider cloud storage (S3, GCS) for very large datasets
4. Check network connection speed

### Directory Not Found

**Error:** `No OptimalWeights directory found on remote`

**Solution:** This is normal if:
- First time training (no weights yet)
- Training hasn't completed
- Use `--skip-weights` or `--data-only` for initial upload

## Advanced Usage

### Custom SSH Configuration

If using non-standard SSH configuration:

```bash
# .env
CLOUD_PORT=2222                    # Non-standard SSH port
CLOUD_SSH_KEY=~/.ssh/custom_key    # Custom key location
```

### Multiple Cloud Instances

Manage multiple instances with separate `.env` files:

```bash
# Development instance
cp .env .env.dev
# Edit .env.dev with dev instance details

# Production instance
cp .env .env.prod
# Edit .env.prod with prod instance details

# Use specific config
export ENV_FILE=.env.dev
./scripts/cloud/upload_to_cloud.sh
```

### Automated Backups

Set up periodic downloads of training checkpoints:

```bash
# crontab entry (every 6 hours)
0 */6 * * * cd /path/to/cs627 && ./scripts/cloud/download_from_cloud.sh --results-only
```

## See Also

- [CLOUD_DEPLOYMENT.md](../../CLOUD_DEPLOYMENT.md) - Complete cloud deployment guide
- [.env.example](../../.env.example) - Environment variable template
- [Training README](../../src/Training/README.md) - Training documentation
