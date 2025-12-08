# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

CS627 AI research project examining **Semantic-Vector Space (SVS)** for multimodal understanding. Encoders convert modalities (text, audio, image) to a shared 1024-dim vector space; decoders convert back.

**Key Design**: Pretrained Encoder/Decoder + MLP Adapter (100x cheaper than full fine-tuning)

## Quick Reference

### Encoders (`src/Encoders/`)

| Class | File | Model | Status |
|-------|------|-------|--------|
| `Text_to_Vec` | `text/semantic_to_vec.py` | facebook/bart-base | PRODUCTION |
| `Audio_to_Vec` | `audio/waveform_to_vec.py` | openai/whisper-small | PRODUCTION |
| `Tone_to_Vec` | `audio/tone_to_vec.py` | microsoft/wavlm-base | PRODUCTION |
| `Image_to_Vec` | `image/visual_to_vec.py` | nlpconnect/vit-gpt2-image-captioning | PRODUCTION |

```python
from Encoders import Text_to_Vec, Audio_to_Vec, Tone_to_Vec, Image_to_Vec
```

### Decoders (`src/Decoders/`)

| Class | File | Model | Status |
|-------|------|-------|--------|
| `Vec_to_Text` | `text/vec_to_semantic.py` | facebook/bart-base | PRODUCTION |
| `Vec_to_Audio` | `audio/vec_to_waveform.py` | microsoft/speecht5_tts | EXPERIMENTAL |
| `Vec_to_Image` | `image/vec_to_visual.py` | CompVis/stable-diffusion-v1-4 | EXPERIMENTAL |

```python
from Decoders import Vec_to_Text, Vec_to_Audio, Vec_to_Image
```

### Adapter (`src/utils/Adapter.py`)

MLP bridge between pretrained models and semantic space.
- Weights saved to: `OptimalWeights/{prefix}_weights.pth`
- Default: 1024 dim, 2 hidden layers, 200 hidden units

## Training (Token-Based)

### New Two-Step Workflow

**Step 1: Pre-generate Tokens** (run once)
```bash
python src/Training/pregenerate_tokens.py
```
Generates HDF5 files with encoder outputs in `data/pregenerated_tokens/`

**Step 2: Train Adapters** (10-100x faster)
```bash
# Train encoder adapters (contrastive learning)
python src/Training/train_adapters.py --mode encoder

# Train decoder adapters (reconstruction)
python src/Training/train_adapters.py --mode decoder
```

### Data Preparation

```bash
# Download MOSI data (first time only)
SKIP_DOWNLOAD=0 python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"
```

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_DOWNLOAD` | `1` | Skip SDK downloads (assume pre-staged data) |
| `MOSI_DATA_PATH` | `data/cmumosi/mosi/` | Path to MOSI metadata |
| `AUDIO_DIR` | `data/cmumosi/audio/` | Extracted audio files |
| `VIDEO_DIR` | `data/cmumosi/frames/` | Extracted video frames |
| `BATCH_SIZE` | `256` | Training batch size (larger with tokens) |
| `ADAPTER_HIDDEN_SIZE` | `512` | Adapter MLP hidden units |
| `USE_AMP` | `1` | Use mixed precision training |

### Performance Benefits
- **10-100x faster**: No encoder forward passes during training
- **6x less memory**: Only adapter parameters in GPU
- **8x larger batches**: Tokens are compact
- **Rapid experiments**: Minutes instead of hours

## Naming Conventions

| Type | File Pattern | Class Pattern |
|------|--------------|---------------|
| Encoder | `{feature}_to_vec.py` | `{Modality}_to_Vec` |
| Decoder | `vec_to_{feature}.py` | `Vec_to_{Modality}` |

## Key Files

```
src/
├── Encoders/
│   ├── encoder_boilerplate.py      # Template for new encoders
│   ├── text/semantic_to_vec.py     # Text_to_Vec
│   ├── audio/waveform_to_vec.py    # Audio_to_Vec
│   ├── audio/tone_to_vec.py        # Tone_to_Vec
│   └── image/visual_to_vec.py      # Image_to_Vec
├── Decoders/
│   ├── decoder_boilerplate.py      # Template for new decoders
│   ├── text/vec_to_semantic.py     # Vec_to_Text
│   ├── audio/vec_to_waveform.py    # Vec_to_Audio
│   └── image/vec_to_visual.py      # Vec_to_Image
├── Training/
│   ├── pregenerate_tokens.py       # Generate tokens once
│   ├── train_adapters.py           # Train adapters only
│   ├── adapter_trainer.py          # Lightweight trainer
│   └── Data_Wrangling/
│       ├── token_dataset.py        # Load pre-generated tokens
│       └── mosi_dataset.py         # Raw data for pre-generation
└── utils/
    └── Adapter.py                  # MLP adapter class
```

## Tools

- `tools/create_encoder.py` - Generate new encoder from template
- `tools/create_decoder.py` - Generate new decoder from template
- `tools/validate_module.py` - Validate production modules

## Important Notes

- All encoders output shape: `(batch_size, 1024)`
- All components inherit from `torch.nn.Module`
- BART text encoder is ground truth for semantic space
- New modules marked EXPERIMENTAL until validated
- Device auto-detected: `"cuda" if torch.cuda.is_available() else "cpu"`

## Documentation Links

- [README.md](README.md) - Setup, quick start, architecture
- [AWS_SETUP.md](AWS_SETUP.md) - Cloud training configuration
- [EVALUATION.md](EVALUATION.md) - Metrics and evaluation guide
- [BENCHMARKS.md](BENCHMARKS.md) - Dataset and benchmark details
- [DOCKER.md](DOCKER.md) - Container setup
- [src/Training/README.md](src/Training/README.md) - Training details
- [src/Encoders/README.md](src/Encoders/README.md) - Encoder guide
- [src/Decoders/README.md](src/Decoders/README.md) - Decoder guide
