# Comprehensive Training Guide

This guide covers the complete token-based training workflow for the CS627 Semantic-Vector Space project.

## Overview

The new training paradigm uses a two-step approach:
1. **Pre-generate tokens** from frozen encoders (one-time cost)
2. **Train adapters** using tokens (10-100x faster iteration)

This approach dramatically reduces training time and resource requirements while maintaining model quality.

## Prerequisites

### Local Setup
- Python 3.8+
- PyTorch 2.0+ (stable version, not 2.9.0)
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- 50GB+ free disk space

### Docker Setup (Recommended)
- Docker 20.10+
- Docker Compose 1.29+
- NVIDIA Docker runtime (for GPU)

## Step 1: Data Preparation

### Download CMU-MOSI Dataset

```bash
# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install .

# Download MOSI metadata
python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"
```

### Extract Audio and Video

```bash
# Download and extract test videos
python scripts/data_wrangling/download_test_videos.py
python scripts/data_wrangling/extract_test_segments.py
```

## Step 2: Token Pre-generation

Pre-generate encoder outputs for all training data. This step runs once and creates reusable tokens.

### Using Docker (Recommended)

```bash
# Auto-detect GPU and generate tokens
./docker/pregenerate.sh

# Or use docker-compose
docker-compose up pregenerate-tokens

# Force CPU mode if needed
./docker/pregenerate.sh --cpu --batch-size 16
```

### Local Execution

```bash
# Generate tokens with default settings
python src/Training/pregenerate_tokens.py

# Custom configuration
BATCH_SIZE=16 DEVICE=cpu python src/Training/pregenerate_tokens.py
```

### Output

Tokens are saved to `data/pregenerated_tokens/mosi/`:
- `train_tokens.h5` - Training set (70%)
- `val_tokens.h5` - Validation set (15%)
- `test_tokens.h5` - Test set (15%)
- `metadata.json` - Token statistics

Each HDF5 file contains:
- Encoder outputs for each modality (1024-dim vectors)
- Segment IDs and labels
- Compressed for efficient storage (~100MB per file)

## Step 3: Adapter Training

Train lightweight MLP adapters using pre-generated tokens.

### Using Docker

```bash
# Train encoder adapters
./docker/train-adapters.sh --mode encoder

# Train decoder adapters
./docker/train-adapters.sh --mode decoder

# Custom settings
./docker/train-adapters.sh --batch-size 512 --epochs 100 --lr 0.0001

# Use docker-compose
docker-compose up train-adapters-gpu
```

### Local Execution

```bash
# Train encoder adapters (contrastive learning)
python src/Training/train_adapters.py --mode encoder

# Train decoder adapters (reconstruction)
python src/Training/train_adapters.py --mode decoder

# Custom hyperparameters
python src/Training/train_adapters.py \
  --mode encoder \
  --epochs 100 \
  --batch-size 512 \
  --lr 0.0001
```

### Training Modes

**Encoder Mode** (Contrastive Learning):
- Aligns different modalities in semantic space
- Uses InfoNCE loss with temperature scaling
- Outputs: Adapter weights for each encoder

**Decoder Mode** (Reconstruction):
- Learns to reconstruct modality features from semantic vectors
- Uses MSE + cosine similarity loss
- Outputs: Adapter weights for each decoder

## Step 4: Ablation Studies

Run systematic experiments to find optimal adapter configurations.

### Using Docker

```bash
# Run recommended configurations
docker-compose up ablation-study

# Or use command line
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/Results:/workspace/Results \
  watsonwb/cs627-svs:gpu \
  python src/Experiments/run_ablation.py --config recommended
```

### Local Execution

```bash
# Test recommended configurations
python src/Experiments/run_ablation.py --config recommended

# Full grid search
python src/Experiments/run_ablation.py --config grid_search

# Custom configuration file
python src/Experiments/run_ablation.py --config-file configs/custom.json
```

### Available Studies

- `recommended` - Best known configurations
- `grid_search` - All combinations of hyperparameters
- `random_search` - Random sampling of configurations
- `progressive_depth` - Varying number of layers
- `activation_study` - Different activation functions
- `dropout_study` - Dropout rate optimization

## Performance Optimization

### GPU Acceleration

```bash
# Enable mixed precision training (2x speedup)
USE_AMP=1 python src/Training/train_adapters.py

# Larger batch sizes for GPU
BATCH_SIZE=512 python src/Training/train_adapters.py
```

### Multi-GPU Training

```bash
# Use all available GPUs
docker run --gpus all ...

# Specific GPUs
docker run --gpus '"device=0,1"' ...
```

### Memory Optimization

```bash
# Gradient accumulation for larger effective batch size
GRADIENT_ACCUMULATION=4 BATCH_SIZE=64 python src/Training/train_adapters.py

# Reduce token generation batch size
BATCH_SIZE=8 python src/Training/pregenerate_tokens.py
```

## Troubleshooting

### PyTorch Segmentation Fault

**Issue**: Python crashes with "Segmentation fault: 11"

**Solution**: Use Docker or downgrade PyTorch:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### No Tokens Found

**Issue**: Adapter training fails with "No pre-generated tokens found"

**Solution**: Run token pre-generation first:
```bash
./docker/pregenerate.sh
```

### CUDA Out of Memory

**Issue**: GPU memory error during training

**Solutions**:
1. Reduce batch size: `BATCH_SIZE=16`
2. Use CPU mode: `DEVICE=cpu`
3. Enable gradient accumulation
4. Use mixed precision: `USE_AMP=1`

### Dataset Not Found

**Issue**: "Dataset files not found" error

**Solution**: Download and extract MOSI data:
```bash
# Download metadata
SKIP_DOWNLOAD=0 python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"

# Extract segments
python scripts/data_wrangling/extract_all_segments.py
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda`/`cpu` | Training device |
| `BATCH_SIZE` | `32` | Token generation batch size |
| `ADAPTER_BATCH_SIZE` | `256` | Adapter training batch size |
| `LEARNING_RATE` | `0.001` | Learning rate |
| `EPOCHS` | `50` | Training epochs |
| `ADAPTER_HIDDEN_SIZE` | `512` | MLP hidden units |
| `ADAPTER_LAYERS` | `2` | Number of hidden layers |
| `USE_AMP` | `1` | Mixed precision training |
| `TOKEN_DIR` | `data/pregenerated_tokens/mosi` | Token location |

### File Locations

| File/Directory | Description |
|---------------|-------------|
| `data/pregenerated_tokens/` | Pre-generated tokens |
| `OptimalWeights/` | Trained adapter weights |
| `Results/adapter_training/` | Training metrics |
| `Results/ablation/` | Ablation study results |
| `configs/ablation/` | Ablation configurations |

## Best Practices

1. **Always use Docker** for reproducible environments
2. **Pre-generate tokens once**, reuse for all experiments
3. **Start with recommended configurations** from ablation studies
4. **Use larger batch sizes** for adapter training (256-512)
5. **Enable mixed precision** on GPUs with Tensor Cores
6. **Monitor validation metrics** to prevent overfitting
7. **Save checkpoints regularly** for long training runs

## Performance Benchmarks

| Metric | Old Pipeline | Token-Based | Improvement |
|--------|-------------|-------------|-------------|
| Token Generation | - | 30 min (once) | - |
| Adapter Training | 3 hours | 15 min | **12x faster** |
| GPU Memory | 12GB | 2GB | **6x reduction** |
| Batch Size | 32 | 256 | **8x larger** |
| Iteration Time | 2-3 hours | 10-15 min | **10x faster** |

## Next Steps

1. Run ablation studies to find optimal adapter architecture
2. Fine-tune on downstream tasks
3. Evaluate on standard benchmarks
4. Deploy trained models

For more details, see:
- [Training README](../src/Training/README.md)
- [Docker Guide](../DOCKER.md)
- [AWS Setup](../AWS_SETUP.md)
- [Evaluation Guide](../EVALUATION.md)