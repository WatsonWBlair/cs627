# CS627 Training Guide

Comprehensive training guide for the Semantic-Vector Space (SVS) project.

## Overview

The CS627 project uses a **two-phase token-based training** approach that is 10-100x faster than traditional end-to-end training:

1. **Pre-generate Tokens**: One-time encoder processing (~30 min)
2. **Train Adapters**: Fast lightweight MLP training (~15 min)

## Quick Start

```bash
# Step 1: Generate tokens (run once)
python src/Training/pregenerate_tokens.py

# Step 2: Train encoder adapters
python src/Training/train_adapters.py --mode encoder

# Step 3: Train decoder adapters
python src/Training/train_adapters.py --mode decoder
```

## Phase 1: Token Pre-generation

### What It Does
Processes raw multimodal data through frozen pre-trained encoders to generate reusable token representations.

### Command Options

```bash
# Basic usage
python src/Training/pregenerate_tokens.py

# Custom batch size for limited memory
BATCH_SIZE=8 python src/Training/pregenerate_tokens.py

# Specify data paths
MOSI_DATA_PATH=/path/to/mosi \
AUDIO_DIR=/path/to/audio \
VIDEO_DIR=/path/to/frames \
python src/Training/pregenerate_tokens.py
```

### Batch Size Recommendations

| GPU VRAM | Batch Size | Time | Memory Usage |
|----------|------------|------|--------------|
| CPU only | 4-8 | ~2 hours | 8GB RAM |
| 8GB | 16 | ~45 min | 6GB VRAM |
| 16GB | 32 | ~30 min | 10GB VRAM |
| 24GB+ | 64 | ~20 min | 14GB VRAM |

### Output Files

Generated tokens are saved to `data/pregenerated_tokens/mosi/`:
- `train_tokens.h5` - 1547 samples (~1.5GB)
- `val_tokens.h5` - 331 samples (~300MB)
- `test_tokens.h5` - 331 samples (~300MB)
- `metadata.json` - Statistics and configuration

### Encoder Models Used

| Modality | Model | Parameters | Token Dim |
|----------|-------|------------|-----------|
| Text | facebook/bart-base | 140M | 1024 |
| Audio | openai/whisper-small | 244M | 1024 |
| Tone | microsoft/wavlm-base | 95M | 1024 |
| Image | nlpconnect/vit-gpt2 | 140M | 1024 |

## Phase 2: Adapter Training

### Architecture
Lightweight MLP adapters map pre-generated tokens to/from the shared semantic space:
```
Tokens (1024d) → [Adapter MLP] → Semantic Space (1024d)
```

### Encoder Adapter Training (Contrastive Learning)

```bash
# Basic training
python src/Training/train_adapters.py --mode encoder

# Custom parameters
python src/Training/train_adapters.py \
    --mode encoder \
    --epochs 100 \
    --batch-size 512 \
    --lr 0.0005 \
    --hidden-size 1024 \
    --num-layers 3
```

### Decoder Adapter Training (Reconstruction)

```bash
# Basic training
python src/Training/train_adapters.py --mode decoder

# Custom parameters
python src/Training/train_adapters.py \
    --mode decoder \
    --epochs 50 \
    --batch-size 256 \
    --lr 0.001
```

### Training Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--batch-size` | 256 | 128-512 | Larger = faster training |
| `--lr` | 0.001 | 0.0001-0.01 | Learning rate |
| `--epochs` | 50 | 30-100 | Training epochs |
| `--hidden-size` | 512 | 256-1024 | Adapter capacity |
| `--num-layers` | 2 | 1-4 | Adapter depth |
| `--dropout` | 0.1 | 0.0-0.3 | Regularization |

### Environment Variables

```bash
# Core configuration
BATCH_SIZE=256              # Training batch size
LEARNING_RATE=0.001         # Optimizer learning rate
EPOCHS=50                   # Number of epochs
ADAPTER_HIDDEN_SIZE=512     # MLP hidden units
ADAPTER_LAYERS=2            # MLP depth

# Performance optimization
USE_AMP=1                   # Mixed precision training
GRADIENT_ACCUMULATION=1     # Gradient accumulation steps
CACHE_SIZE=2000            # Token cache size

# Data paths
MOSI_DATA_PATH=data/cmumosi/mosi/
AUDIO_DIR=data/cmumosi/audio/
VIDEO_DIR=data/cmumosi/frames/
OUTPUT_DIR=data/pregenerated_tokens/
```

## Performance Comparison

| Metric | Traditional | Token-Based | Improvement |
|--------|------------|-------------|-------------|
| **Parameters** | 619M (all) | 7.7M (adapters) | 99% reduction |
| **Training Speed** | ~1 epoch/min | ~10 epochs/min | 10x faster |
| **GPU Memory** | 12GB | 2GB | 6x reduction |
| **Batch Size** | 32 | 256 | 8x larger |
| **Total Time** | 2-3 hours | 10-15 minutes | 10x faster |
| **Cost** | ~$50 | ~$5 | 10x cheaper |

## Advanced Training Techniques

### Mixed Precision Training
```bash
# Enable automatic mixed precision for 2x speedup
USE_AMP=1 python src/Training/train_adapters.py --mode encoder
```

### Gradient Accumulation
```bash
# Simulate larger batch sizes on limited memory
GRADIENT_ACCUMULATION=4 BATCH_SIZE=64 python src/Training/train_adapters.py
```

### Learning Rate Scheduling
```bash
# Cosine annealing with warmup
python src/Training/train_adapters.py \
    --scheduler cosine \
    --warmup-epochs 5
```

### Multi-GPU Training
```bash
# Use multiple GPUs (DataParallel)
CUDA_VISIBLE_DEVICES=0,1 python src/Training/train_adapters.py
```

## Ablation Studies

### Run Pre-configured Studies

```bash
# Recommended configurations (3 best configs)
python src/Experiments/run_ablation.py --config recommended

# Full grid search (192 configs)
python src/Experiments/run_ablation.py --config grid_search

# Random search (20 configs)
python src/Experiments/run_ablation.py --config random_search
```

### Custom Ablation Configuration

Create `configs/ablation/custom.json`:
```json
{
  "configs": [
    {"hidden_size": 256, "num_layers": 2, "activation": "relu"},
    {"hidden_size": 512, "num_layers": 3, "activation": "gelu"},
    {"hidden_size": 1024, "num_layers": 2, "activation": "swish"}
  ],
  "epochs": 30,
  "batch_size": 256
}
```

Run custom study:
```bash
python src/Experiments/run_ablation.py --config-file configs/ablation/custom.json
```

## Expected Training Results

### Encoder Adapters (Contrastive)
- **Loss**: 2.0 → 0.3-0.5
- **Recall@1**: 10% → 50-70%
- **Recall@5**: 30% → 80-90%
- **Convergence**: 20-30 epochs

### Decoder Adapters (Reconstruction)
- **MSE Loss**: 1.0 → 0.05-0.1
- **Cosine Similarity**: 0.5 → 0.85-0.95
- **Convergence**: 30-40 epochs

## Output Files

### Adapter Weights
Saved to `OptimalWeights/`:
- `text_adapter_weights.pth` - Text encoder adapter
- `audio_adapter_weights.pth` - Audio encoder adapter
- `tone_adapter_weights.pth` - Tone encoder adapter
- `image_adapter_weights.pth` - Image encoder adapter
- `text_decoder_adapter_weights.pth` - Text decoder adapter
- `audio_decoder_adapter_weights.pth` - Audio decoder adapter
- `image_decoder_adapter_weights.pth` - Image decoder adapter

### Training History
Saved to `Results/adapter_training/`:
- `encoder_training_YYYYMMDD_HHMMSS.json` - Encoder metrics
- `decoder_training_YYYYMMDD_HHMMSS.json` - Decoder metrics
- `ablation_YYYYMMDD_HHMMSS.json` - Ablation results

## Monitoring Training

### Real-time Progress
```bash
# Watch training logs
tail -f Results/adapter_training/latest.log

# Monitor GPU usage
nvidia-smi -l 1

# Check token generation progress
watch -n 1 ls -lh data/pregenerated_tokens/mosi/
```

### TensorBoard (if configured)
```bash
# Launch TensorBoard
tensorboard --logdir Results/tensorboard

# Access at http://localhost:6006
```

## Common Issues & Solutions

### Memory Issues

**Problem**: CUDA out of memory
```bash
# Solution 1: Reduce batch size
BATCH_SIZE=128 python src/Training/train_adapters.py

# Solution 2: Enable gradient accumulation
GRADIENT_ACCUMULATION=4 BATCH_SIZE=64 python src/Training/train_adapters.py

# Solution 3: Use mixed precision
USE_AMP=1 python src/Training/train_adapters.py
```

### Training Issues

**Problem**: Loss not decreasing
```bash
# Solution 1: Lower learning rate
python src/Training/train_adapters.py --lr 0.0001

# Solution 2: Increase model capacity
ADAPTER_HIDDEN_SIZE=1024 ADAPTER_LAYERS=3 python src/Training/train_adapters.py

# Solution 3: Check data normalization
python -c "import h5py; f=h5py.File('data/pregenerated_tokens/mosi/train_tokens.h5'); 
print('Mean:', f['text_base'][:].mean(), 'Std:', f['text_base'][:].std())"
```

**Problem**: Validation metrics plateauing
```bash
# Solution 1: Add dropout regularization
python src/Training/train_adapters.py --dropout 0.2

# Solution 2: Use learning rate scheduling
python src/Training/train_adapters.py --scheduler cosine

# Solution 3: Early stopping
python src/Training/train_adapters.py --early-stopping-patience 10
```

### Data Issues

**Problem**: No tokens found
```bash
# Solution: Generate tokens first
python src/Training/pregenerate_tokens.py

# Verify tokens exist
ls -lh data/pregenerated_tokens/mosi/
```

**Problem**: MOSI dataset not found
```bash
# Solution: Download MOSI data
SKIP_DOWNLOAD=0 python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; 
download_mosi('data/cmumosi/mosi/')"
```

## Best Practices

### For Speed
1. Use largest batch size that fits in memory
2. Enable mixed precision (`USE_AMP=1`)
3. Pre-generate tokens once, reuse for all experiments
4. Use gradient accumulation for effective larger batches

### For Quality
1. Start with default hyperparameters
2. Use validation-based early stopping
3. Run multiple seeds and ensemble
4. Monitor both loss and task metrics

### For Reproducibility
1. Set random seeds explicitly
2. Save configuration with results
3. Version control adapter weights
4. Document environment and dependencies

## Next Steps

1. **Evaluate Models**: See [EVALUATION.md](EVALUATION.md)
2. **Run Inference**: See [src/Inference/README.md](src/Inference/README.md)
3. **Deploy to Cloud**: See [AWS_SETUP.md](AWS_SETUP.md)
4. **Optimize Further**: See [BENCHMARKS.md](BENCHMARKS.md)

## Related Documentation

- [README.md](README.md) - Project overview
- [CLAUDE.md](CLAUDE.md) - AI assistant guidance
- [COSTS.md](COSTS.md) - Cost breakdown
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Extended troubleshooting