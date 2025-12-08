# Training Guide

Complete training workflow for the CS627 Semantic-Vector Space project.

## Overview

Training uses a two-step approach:
1. **Pre-generate tokens** from frozen encoders (one-time, ~30 min)
2. **Train adapters** using tokens (fast iteration, ~15 min)

This approach is 10-100x faster than end-to-end training.

## Step 1: Token Pre-generation

### Basic Usage

```bash
# Using Make
make tokens

# Using Docker
docker-compose up pregenerate-tokens

# Direct Python
python src/Training/pregenerate_tokens.py
```

### Batch Size Recommendations

| GPU VRAM | Batch Size | Time |
|----------|------------|------|
| CPU only | 4-8 | ~2 hours |
| 8GB | 16 | ~45 min |
| 16GB | 32 | ~30 min |
| 24GB+ | 64 | ~20 min |

### Custom Configuration

```bash
# Small batch for limited memory
BATCH_SIZE=8 python src/Training/pregenerate_tokens.py

# Specific data paths
MOSI_DATA_PATH=/path/to/mosi \
AUDIO_DIR=/path/to/audio \
VIDEO_DIR=/path/to/frames \
python src/Training/pregenerate_tokens.py
```

### Output Files

Tokens saved to `data/pregenerated_tokens/mosi/`:
- `train_tokens.h5` - 1547 samples (~1.5GB)
- `val_tokens.h5` - 331 samples (~300MB)  
- `test_tokens.h5` - 331 samples (~300MB)
- `metadata.json` - Statistics and configuration

## Step 2: Adapter Training

### Encoder Adapters (Contrastive Learning)

```bash
# Using Make
make train

# Using Docker
docker-compose up train-adapters-gpu

# Direct Python
python src/Training/train_adapters.py --mode encoder
```

### Decoder Adapters (Reconstruction)

```bash
# Train decoders
python src/Training/train_adapters.py --mode decoder

# With custom parameters
python src/Training/train_adapters.py \
    --mode decoder \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.0005
```

### Training Parameters

| Parameter | Default | Recommended Range |
|-----------|---------|-------------------|
| Batch Size | 256 | 128-512 |
| Learning Rate | 0.001 | 0.0001-0.01 |
| Epochs | 50 | 30-100 |
| Hidden Size | 512 | 256-1024 |
| Layers | 2 | 1-4 |

### Memory Requirements

| Mode | GPU Memory | Batch Size |
|------|------------|------------|
| Token Generation | 8-12GB | 16-32 |
| Adapter Training | 2-4GB | 256-512 |
| CPU Training | RAM 8GB+ | 64-128 |

## Step 3: Ablation Studies

### Run Recommended Configurations

```bash
# Using CLI
python cli.py ablation --configs recommended

# Direct Python
python src/Experiments/run_ablation.py --config recommended
```

### Available Configurations

1. **recommended** - 3 best configs from preliminary testing
2. **grid_search** - Full parameter sweep (192 configs)
3. **random_search** - Random sampling (20 configs)
4. **progressive_depth** - Test layer depth impact
5. **activation_study** - Compare activation functions
6. **dropout_study** - Test dropout rates

### Custom Ablation Config

Create `configs/ablation/custom.json`:
```json
{
  "configs": [
    {
      "hidden_size": 256,
      "num_layers": 2,
      "activation": "relu",
      "dropout": 0.1
    },
    {
      "hidden_size": 512,
      "num_layers": 3,
      "activation": "gelu",
      "dropout": 0.2
    }
  ],
  "epochs": 30,
  "batch_size": 256
}
```

Run custom config:
```bash
python src/Experiments/run_ablation.py --config-file configs/ablation/custom.json
```

### Interpreting Results

Results saved to `Results/ablation/`:
- `ablation_YYYYMMDD_HHMMSS.json` - All metrics
- `best_config.json` - Best performing configuration

Key metrics to evaluate:
- **val_loss** - Lower is better (< 0.5 good)
- **recall@1** - Higher is better (> 50% good)
- **recall@5** - Higher is better (> 80% good)

## Advanced Training

### Multi-GPU Training

```bash
# Using DataParallel (single node)
CUDA_VISIBLE_DEVICES=0,1 python src/Training/train_adapters.py

# Note: Currently single-GPU optimized
```

### Mixed Precision Training

```bash
# Enable AMP (automatic mixed precision)
USE_AMP=1 python src/Training/train_adapters.py
```

### Gradient Accumulation

```bash
# Simulate larger batch size
GRADIENT_ACCUMULATION=4 \
BATCH_SIZE=64 \
python src/Training/train_adapters.py
```

### Resume Training

```bash
# Resume from checkpoint (not yet implemented)
python src/Training/train_adapters.py --resume OptimalWeights/checkpoint.pth
```

## Monitoring Training

### View Progress

```bash
# Check training status
make status

# Monitor logs
tail -f Results/adapter_training/latest.log

# Using TensorBoard (if configured)
tensorboard --logdir Results/tensorboard
```

### Expected Training Curves

Encoder adapters (contrastive):
- Loss: 2.0 → 0.3-0.5
- Recall@1: 10% → 50-70%
- Recall@5: 30% → 80-90%

Decoder adapters (reconstruction):
- MSE Loss: 1.0 → 0.05-0.1
- Cosine Similarity: 0.5 → 0.85-0.95

## Optimizing Performance

### Speed Optimization

1. **Increase batch size**: Tokens are small, use 256-512
2. **Enable AMP**: `USE_AMP=1` for mixed precision
3. **Use gradient accumulation**: Simulate larger batches
4. **Pre-load to RAM**: Set cache_size in token dataset

### Quality Optimization

1. **Learning rate schedule**: Start with 0.001, decay by 0.1
2. **Early stopping**: Stop if val_loss doesn't improve for 10 epochs
3. **Ensemble**: Train multiple seeds, average predictions
4. **Data augmentation**: Add noise to tokens during training

## Troubleshooting

### "No tokens found"
```bash
# Generate tokens first
make tokens
```

### "CUDA out of memory"
```bash
# Reduce batch size
BATCH_SIZE=128 python src/Training/train_adapters.py
```

### "Training loss not decreasing"
```bash
# Lower learning rate
python src/Training/train_adapters.py --lr 0.0001

# Check data normalization
python -c "import h5py; f=h5py.File('data/pregenerated_tokens/mosi/train_tokens.h5'); print(f['text_base'][:].mean(), f['text_base'][:].std())"
```

### "Val metrics plateauing early"
```bash
# Increase model capacity
ADAPTER_HIDDEN_SIZE=1024 ADAPTER_LAYERS=3 python src/Training/train_adapters.py

# Add dropout
python src/Training/train_adapters.py --dropout 0.2
```

## Output Files

Trained weights saved to `OptimalWeights/`:
- `text_adapter_weights.pth`
- `audio_adapter_weights.pth`
- `tone_adapter_weights.pth`
- `image_adapter_weights.pth`
- `text_decoder_adapter_weights.pth` (decoder mode)

Training history saved to `Results/adapter_training/`:
- `encoder_training_YYYYMMDD_HHMMSS.json`
- `decoder_training_YYYYMMDD_HHMMSS.json`

## Next Steps

1. **Evaluate models**: `make evaluate`
2. **Run inference**: See [Inference Guide](../src/Inference/README.md)
3. **Deploy models**: See [AWS_SETUP.md](../AWS_SETUP.md)