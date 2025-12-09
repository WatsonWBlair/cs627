# Training Pipeline

This directory contains the token-based training pipeline for the Semantic Vector Space (SVS) project. The new approach pre-generates encoder tokens once, then trains only lightweight adapter MLPs, achieving 10-100x faster training.

## Quick Start

### Step 1: Pre-generate Tokens

Generate encoder tokens from raw multimodal data:

```bash
python src/Training/pregenerate_tokens.py
```

This creates HDF5 files with pre-computed tokens in `data/pregenerated_tokens/mosi/`:
- `train_tokens.h5` - Training set tokens
- `val_tokens.h5` - Validation set tokens  
- `test_tokens.h5` - Test set tokens
- `metadata.json` - Token statistics

### Step 2: Train Adapters

Train adapter MLPs using pre-generated tokens:

```bash
# Train encoder adapters (contrastive learning)
python src/Training/train_adapters.py --mode encoder

# Train decoder adapters (reconstruction)
python src/Training/train_adapters.py --mode decoder
```

## Architecture

```
Raw Data → [Frozen Encoders] → Tokens → [Trainable Adapters] → Semantic Space
```

### Key Components

1. **Token Pre-generation** (`pregenerate_tokens.py`)
   - Processes raw data through frozen encoders with `pregen=True`
   - Saves tokens to compressed HDF5 format
   - One-time cost, reusable across experiments

2. **Token Dataset** (`Data_Wrangling/token_dataset.py`)
   - Efficient HDF5-based data loading
   - Memory-mapped access for large datasets
   - Optional caching for frequently used samples

3. **Adapter Training** (`train_adapters.py`)
   - Trains only adapter MLPs (not encoders)
   - Supports contrastive and reconstruction objectives
   - Mixed precision and gradient accumulation

4. **Lightweight Trainer** (`adapter_trainer.py`)
   - Simplified trainer for token-based training
   - No encoder forward passes during training
   - Optimized for adapter parameters only

## Performance Benefits

| Metric | Old Approach | Token-Based | Improvement |
|--------|-------------|-------------|-------------|
| Training Speed | ~1 epoch/min | ~10 epochs/min | **10x faster** |
| GPU Memory | 12GB | 2GB | **6x reduction** |
| Batch Size | 32 | 256 | **8x larger** |
| Experiment Time | 2-3 hours | 10-15 minutes | **10x faster** |

## Configuration

### Environment Variables

See [docs/ENVIRONMENT_VARIABLES.md](../../docs/ENVIRONMENT_VARIABLES.md) for the complete reference.

Key variables:
```bash
BATCH_SIZE=256           # Training batch size (larger with tokens)
LEARNING_RATE=0.001      # Optimizer learning rate
EPOCHS=50                # Training epochs
USE_AMP=1                # Mixed precision
```

### Training Modes

- **Encoder Mode**: Trains adapters to map encoder features to semantic space using contrastive learning
- **Decoder Mode**: Trains adapters to reconstruct modality features from semantic vectors

## Data Flow

1. **Raw Data**: Text, audio waveforms, video frames from CMU-MOSI
2. **Encoders**: Pre-trained models (BART, Whisper, WavLM, ViT)
3. **Tokens**: Fixed feature representations (1024-dim)
4. **Adapters**: Lightweight MLPs (2-4 layers, 128-512 hidden units)
5. **Semantic Space**: Shared 1024-dimensional representation

## File Structure

```
src/Training/
├── pregenerate_tokens.py      # Generate tokens once
├── train_adapters.py           # Train adapters only
├── adapter_trainer.py          # Lightweight trainer
├── Data_Wrangling/
│   ├── token_dataset.py       # Load pre-generated tokens
│   └── mosi_dataset.py        # Raw data access (for pre-generation)
└── Experiments/                # Ablation studies (upcoming)
```

## Dataset: CMU-MOSI

- 2,199 video segments with aligned text, audio, video
- Auto-downloaded via CMU-MultimodalSDK
- Split: 70% train, 15% val, 15% test

### Installation

```bash
# Install CMU-MultimodalSDK (for MOSI data)
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..
```

## Advanced Usage

### Custom Token Sources

```python
from src.Training.Data_Wrangling.token_dataset import TokenDataset

# Load specific encoders
dataset = TokenDataset(
    h5_path='data/pregenerated_tokens/mosi/train_tokens.h5',
    encoders=['text_base', 'audio_waveform'],
    token_mode='matched',
    cache_size=2000
)
```

### Ablation Studies

Coming soon: Automated ablation studies for adapter configurations.

```bash
# Run ablation study (upcoming)
python src/Experiments/run_ablation.py --config configs/ablation.yaml
```

## Output

Trained adapter weights saved to `OptimalWeights/`:
- `text_adapter_weights.pth`
- `audio_adapter_weights.pth`
- `tone_adapter_weights.pth`
- `image_adapter_weights.pth`

## Success Criteria

### Encoder Adapters
- Recall@1 > 50%
- Recall@5 > 80%
- Average cosine similarity > 0.8

### Decoder Adapters
- Reconstruction MSE < 0.1
- Cosine similarity > 0.9
- Semantic fidelity preserved

## GPU Memory Guidelines

| GPU VRAM | Token Gen Batch | Training Batch |
|----------|-----------------|----------------|
| 8 GB | 16 | 128 |
| 16 GB | 32 | 256 |
| 24 GB | 64 | 512 |

## Troubleshooting

### No tokens found
Run `pregenerate_tokens.py` first to generate token files.

### Out of memory during generation
Reduce `BATCH_SIZE` for token generation (doesn't affect training speed).

### Slow training
- Increase `BATCH_SIZE` for adapter training (tokens are small)
- Enable mixed precision: `USE_AMP=1`
- Use gradient accumulation for larger effective batch sizes

## Migration from Old Training

The old end-to-end training scripts have been removed in favor of the token-based approach:

- ~~`train_encoders.py`~~ → Use `pregenerate_tokens.py` + `train_adapters.py --mode encoder`
- ~~`train_decoders.py`~~ → Use `pregenerate_tokens.py` + `train_adapters.py --mode decoder`
- ~~`encoder_trainers.py`~~ → Replaced by `adapter_trainer.py`
- ~~`decoder_trainer.py`~~ → Replaced by `adapter_trainer.py`

Existing model weights remain compatible for inference.

## Related Documentation

- [docs/TRAINING_GUIDE.md](../../docs/TRAINING_GUIDE.md) - Training concepts, best practices, expected results
- [docs/LOCAL_TRAINING.md](../../docs/LOCAL_TRAINING.md) - Step-by-step workflow
- [docs/ENVIRONMENT_VARIABLES.md](../../docs/ENVIRONMENT_VARIABLES.md) - All configuration options
- [src/Experiments/README.md](../Experiments/README.md) - Ablation study framework