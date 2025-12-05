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

## Training

### Encoder Training (`src/Training/train_encoders.py`)
- Uses Cross-Modal Momentum Contrastive Learning (MoCo)
- Dataset: CMU-MOSI (text, audio, video segments)
- Trainer: `Contrast` class in `encoder_trainers.py`

### Decoder Training (`src/Training/train_decoders.py`)
- Dual loss: Reconstruction + Semantic Fidelity
- Trainer: `CrossModalDecoderTrainer` in `decoder_trainer.py`

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
│   ├── train_encoders.py           # Main encoder training
│   ├── train_decoders.py           # Main decoder training
│   ├── encoder_trainers.py         # Contrast trainer (MoCo)
│   └── decoder_trainer.py          # CrossModalDecoderTrainer
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
