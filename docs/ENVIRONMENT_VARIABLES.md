# Environment Variables Reference

Centralized reference for all environment variables used in the CS627 project.

## Core Configuration

### Device & Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Device to use (`cuda` or `cpu`) |
| `USE_AMP` | `1` | Enable mixed precision training (0/1) |
| `GRADIENT_ACCUMULATION` | `1` | Gradient accumulation steps |
| `CACHE_SIZE` | `2000` | Token cache size for frequent samples |

### Batch Sizes

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | `32` | Token generation batch size |
| `ADAPTER_BATCH_SIZE` | `256` | Adapter training batch size |

### Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `MOSI_DATA_PATH` | `data/cmumosi/mosi/` | MOSI metadata (.cdf files) |
| `AUDIO_DIR` | `data/cmumosi/audio/` | Extracted audio files |
| `VIDEO_DIR` | `data/cmumosi/frames/` | Extracted video frames |
| `TOKEN_DIR` | `data/pregenerated_tokens/mosi/` | Pre-generated tokens |
| `OUTPUT_DIR` | `data/pregenerated_tokens/` | Token output directory |
| `GLUE_DATA_PATH` | `data/glue/` | GLUE dataset files |
| `MTEB_DATA_PATH` | `data/mteb/` | MTEB dataset files |

### Training Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | `50` | Number of training epochs |
| `LEARNING_RATE` | `0.001` | Optimizer learning rate |
| `ADAPTER_HIDDEN_SIZE` | `512` | Adapter MLP hidden units |
| `ADAPTER_LAYERS` | `2` | Adapter MLP depth |

### Data Wrangling

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_INTERVAL` | `5000` | Samples between checkpoints during data wrangling |

## AWS Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_BUCKET` | `cs627-svs-data` | S3 bucket for cloud storage |
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_DEFAULT_REGION` | `us-east-1` | Default AWS region |

## Quick Reference by Task

### Token Generation

```bash
BATCH_SIZE=32 \
MOSI_DATA_PATH=data/cmumosi/mosi/ \
AUDIO_DIR=data/cmumosi/audio/ \
VIDEO_DIR=data/cmumosi/frames/ \
OUTPUT_DIR=data/pregenerated_tokens/ \
python src/Training/pregenerate_tokens.py
```

### Adapter Training

```bash
BATCH_SIZE=256 \
LEARNING_RATE=0.001 \
EPOCHS=50 \
ADAPTER_HIDDEN_SIZE=512 \
ADAPTER_LAYERS=2 \
USE_AMP=1 \
python src/Training/train_adapters.py --mode encoder
```

### Memory-Constrained Systems

```bash
# Low memory token generation
BATCH_SIZE=8 DEVICE=cpu python src/Training/pregenerate_tokens.py

# Low memory training
BATCH_SIZE=64 GRADIENT_ACCUMULATION=4 python src/Training/train_adapters.py
```

### Docker

```bash
docker run --gpus all \
  -e BATCH_SIZE=256 \
  -e EPOCHS=100 \
  -e LEARNING_RATE=0.0005 \
  -e USE_AMP=1 \
  -v $(pwd)/data:/workspace/data \
  watsonwb/cs627-svs:latest \
  python src/Training/train_adapters.py
```

## Recommended Values by Hardware

### CPU Only

```bash
DEVICE=cpu
BATCH_SIZE=8        # Token generation
ADAPTER_BATCH_SIZE=64
USE_AMP=0
```

### 8GB GPU

```bash
DEVICE=cuda
BATCH_SIZE=16       # Token generation
ADAPTER_BATCH_SIZE=128
USE_AMP=1
```

### 16GB GPU

```bash
DEVICE=cuda
BATCH_SIZE=32       # Token generation
ADAPTER_BATCH_SIZE=256
USE_AMP=1
```

### 24GB+ GPU

```bash
DEVICE=cuda
BATCH_SIZE=64       # Token generation
ADAPTER_BATCH_SIZE=512
USE_AMP=1
```

## Using .env Files

Create a `.env` file in the project root:

```bash
# .env
DEVICE=cuda
BATCH_SIZE=32
ADAPTER_BATCH_SIZE=256
EPOCHS=50
LEARNING_RATE=0.001
ADAPTER_HIDDEN_SIZE=512
ADAPTER_LAYERS=2
USE_AMP=1

# Paths
MOSI_DATA_PATH=data/cmumosi/mosi/
AUDIO_DIR=data/cmumosi/audio/
VIDEO_DIR=data/cmumosi/frames/
TOKEN_DIR=data/pregenerated_tokens/mosi/

# AWS (optional)
S3_BUCKET=my-bucket-name
AWS_REGION=us-east-1
```

Load with:
```bash
export $(cat .env | xargs)
```

## Related Documentation

- [SETUP.md](SETUP.md) - Installation guide
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training workflow
- [DOCKER.md](DOCKER.md) - Container configuration
- [docs/aws/README.md](aws/README.md) - AWS deployment
