# Local Training Guide

Step-by-step training workflow for running token pregeneration and adapter training locally.

**Prerequisites**: See [SETUP.md](SETUP.md) for installation, platform-specific setup, and system requirements.

**Advanced options**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for hyperparameters and optimization.

## Overview

The training workflow has two phases:

1. **Pre-generate Tokens** (~30 min): Process raw data through frozen encoders once
2. **Train Adapters** (~15 min): Train lightweight MLPs on pre-generated tokens

## Training Workflow

### Step 1: Create Directories

```bash
mkdir -p data/cmumosi/{mosi,audio,frames}
mkdir -p data/pregenerated_tokens/mosi
mkdir -p OptimalWeights Results
```

### Step 2: Download Dataset

```bash
# Download CMU-MOSI data
SKIP_DOWNLOAD=0 python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"
```

### Step 3: Pre-generate Tokens

```bash
python src/Training/pregenerate_tokens.py
```

Adjust batch size for your hardware:
```bash
# Limited memory (8GB RAM/VRAM)
BATCH_SIZE=8 python src/Training/pregenerate_tokens.py

# Windows CMD (different env var syntax)
set BATCH_SIZE=8
python src/Training/pregenerate_tokens.py
```

**Expected output** in `data/pregenerated_tokens/mosi/`:
- `train_tokens.h5` (~1.5GB)
- `val_tokens.h5` (~300MB)
- `test_tokens.h5` (~300MB)
- `metadata.json`

### Step 4: Train Encoder Adapters

```bash
python src/Training/train_adapters.py --mode encoder
```

Custom parameters:
```bash
python src/Training/train_adapters.py --mode encoder --epochs 100 --batch-size 512 --lr 0.0005
```

### Step 5: Train Decoder Adapters

```bash
python src/Training/train_adapters.py --mode decoder
```

### Step 6: Verify Results

```bash
ls OptimalWeights/
# Expected: text_adapter_weights.pth, audio_adapter_weights.pth, etc.
```

## Ablation Studies

Run ablation studies to compare adapter configurations. See [src/Experiments/README.md](../src/Experiments/README.md) for full API.

```bash
# Run recommended configurations
python src/Experiments/run_ablation.py --config recommended

# Run full grid search
python src/Experiments/run_ablation.py --config grid_search
```

### Visualize Results

```bash
# Visualize latest results
python scripts/visualize_ablation.py

# Specify results file
python scripts/visualize_ablation.py --results Results/ablation/ablation_results_*.json
```

**Output figures** in `Results/ablation/figures/`.

## Quick Reference

### Direct Python Commands

| Task | Command |
|------|---------|
| Generate tokens | `python src/Training/pregenerate_tokens.py` |
| Train encoder adapters | `python src/Training/train_adapters.py --mode encoder` |
| Train decoder adapters | `python src/Training/train_adapters.py --mode decoder` |
| Run ablation study | `python src/Experiments/run_ablation.py --config recommended` |

### Windows make.bat

| Task | Command |
|------|---------|
| Tokens (no Docker) | `make.bat tokens-local` |
| Tokens (CPU Docker) | `make.bat tokens-cpu` |
| Train (no Docker) | `make.bat train-local` |
| Train (CPU Docker) | `make.bat train-cpu` |
| Ablation | `make.bat ablation-local` |

### Make Commands (Unix)

| Task | Command |
|------|---------|
| Generate tokens | `make tokens` |
| Train adapters | `make train` |
| Run evaluation | `make evaluate` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'mmsdk'` | Install CMU-MultimodalSDK (see [SETUP.md](SETUP.md)) |
| CUDA out of memory | Reduce `BATCH_SIZE` (8 or 16) |
| `make` not found (Windows) | Use `make.bat` or direct Python commands |
| Token files not found | Run `pregenerate_tokens.py` first |
| Docker GPU not working | Use `-local` commands |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for extended solutions.

## Next Steps

- [EVALUATION.md](EVALUATION.md) - Evaluate trained models
- [BENCHMARKS.md](BENCHMARKS.md) - Performance baselines
- [docs/aws/](aws/) - Cloud training
