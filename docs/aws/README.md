# AWS Deployment Guide

Deploy CS627 training on AWS EC2 with GPU support.

## Quick Start

```bash
# 1. Setup AWS resources (one-time)
./scripts/setup/setup_aws_resources.sh --region us-east-1
source .env.cloud

# 2. Launch training instance
./scripts/cloud/launch_training_instance.sh --instance-type g5.4xlarge

# 3. SSH and run training
ssh -i your-key.pem ubuntu@<instance-ip>
cd cs627
python src/Training/pregenerate_tokens.py   # ~30 min
python src/Training/train_adapters.py --mode encoder  # ~10 min
```

## Choose Your Setup Method

| Method | Best For | Guide |
|--------|----------|-------|
| **CLI** | Automation, scripting, CI/CD | [CLI_SETUP.md](CLI_SETUP.md) |
| **Web Console** | First-time users, visual setup | [CONSOLE_SETUP.md](CONSOLE_SETUP.md) |

## Recommended Configuration

| Component | Specification | Cost |
|-----------|--------------|------|
| **Instance** | g5.4xlarge | $1.62/hour |
| **GPU** | NVIDIA A10G (24GB) | Included |
| **Storage** | 100 GB gp3 | $8/month |

**Estimated training cost**: $2-5 for full pipeline (~1 hour total)

## Instance Options

| Instance | GPU | VRAM | Cost/Hour | Use Case |
|----------|-----|------|-----------|----------|
| g4dn.xlarge | T4 | 16GB | $0.53 | Budget training |
| g5.xlarge | A10G | 24GB | $1.01 | Standard training |
| g5.4xlarge | A10G | 24GB | $1.62 | Fast training (recommended) |

## Training Strategies

| Approach | Cloud Time | Cost | Best For |
|----------|------------|------|----------|
| Generate locally, train on cloud | ~20 min | $0.50 | Multiple experiments |
| Full cloud pipeline | ~50 min | $1.35 | Single run |
| Pre-staged tokens on S3 | ~20 min | $0.50/run | Team collaboration |

## Documentation

- [CLI_SETUP.md](CLI_SETUP.md) - Command-line setup (IAM, S3, EC2)
- [CONSOLE_SETUP.md](CONSOLE_SETUP.md) - AWS Console web UI setup
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - AWS-specific issues

## Related

- [docs/TRAINING_GUIDE.md](../TRAINING_GUIDE.md) - Training workflow
- [docs/DOCKER.md](../DOCKER.md) - Container usage
- [docs/COSTS.md](../COSTS.md) - Cost breakdown
