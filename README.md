# Semantic-Vector Space for Multimodal Understanding

CS627 AI Research Project examining the impact of a shared **Semantic-Vector Space (SVS)** on Natural Language Understanding (NLU) tasks.

## Overview

This project implements an encoder-decoder architecture that aligns multiple modalities (text, audio, video) to a shared semantic vector space. By training inference models to operate in this common representation space, we enable:

- **Cross-modal understanding**: Text, audio, and video share the same semantic representation
- **Efficient training**: 100x cheaper than full fine-tuning (only train small MLP adapters)
- **Modular design**: Add new modalities without retraining existing encoders

**Key Innovation**: BERT-style architecture (Pretrained Model + MLP Adapter) trained with Cross-Modal Momentum Contrastive Learning (MoCo).

## Quick Start

### Local Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install CMU-MultimodalSDK for dataset access
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install .

# Download and extract MOSI videos
python scripts/data_wrangling/download_test_videos.py
python scripts/data_wrangling/extract_test_segments.py

# Train text, audio, and video encoders on CMU-MOSI dataset
python src/Training/train_encoders.py
```

See [Training Quick Start](src/Training/QUICKSTART.md) for detailed instructions.

### Cloud GPU Setup (Recommended)

For faster training on cloud GPU instances (AWS, GCP, Azure, Lambda Labs):

```bash
# Clone repository on cloud instance
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627

# Run automated setup and training
chmod +x setup_and_train.sh
./setup_and_train.sh
```

See [Cloud Deployment Guide](CLOUD_DEPLOYMENT.md) for complete cloud setup instructions.

### 3. Use Trained Encoders

```python
from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec

# Load encoders (adapters auto-load trained weights)
text_encoder = Text_to_Vec()
audio_encoder = Audio_to_Vec()

# Encode to shared semantic space
text_vector = text_encoder("This is a test")
audio_vector = audio_encoder(audio_waveform)

# Measure cross-modal similarity
import torch.nn.functional as F
similarity = F.cosine_similarity(text_vector, audio_vector)
```

## Architecture

### Encoders (Input → Semantic Vector)

**Design**: Pretrained Model + MLP Adapter

```
Text/Audio/Video → Pretrained Encoder → MLP Adapter → Semantic Vector (1024-dim)
```

**Current Encoders** (organized by modality):
- **Text** (`text/`): `Text_to_Vec` - `facebook/bart-base` + Adapter → `semantic_to_vec.py` (PRODUCTION)
- **Audio** (`audio/`): `Audio_to_Vec` - `openai/whisper-small` + Adapter → `waveform_to_vec.py` (PRODUCTION)
- **Image** (`image/`): `Image_to_Vec` - `nlpconnect/vit-gpt2-image-captioning` + Adapter → `visual_to_vec.py` (PRODUCTION)

Import: `from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec`

See [Encoder README](src/Encoders/README.md) for adding new encoders.

### Decoders (Semantic Vector → Output)

**Design**: MLP Adapter + Pretrained Decoder

```
Semantic Vector → MLP Adapter → Pretrained Decoder → Text/Audio/Image
```

**Current Decoders** (organized by modality):
- **Text** (`text/`): `Vec_to_Text` - Adapter + `facebook/bart-base` → `vec_to_semantic.py` (PRODUCTION)
- **Image** (`image/`): `Vec_to_Image` - Adapter + `CompVis/stable-diffusion-v1-4` → `vec_to_visual.py` (EXPERIMENTAL)
- **Audio** (`audio/`): `Vec_to_Audio` - Adapter + `suno/bark-small` → `vec_to_waveform.py` (EXPERIMENTAL)

Import: `from Decoders import Vec_to_Text, Vec_to_Audio, Vec_to_Image`

See [Decoder README](src/Decoders/README.md) for adding new decoders.

### Training

**Encoder Alignment**: Cross-Modal Momentum Contrastive Learning (MoCo)
- Momentum encoder with exponential moving average
- Memory queue for large-scale negative sampling
- InfoNCE loss for stable contrastive learning

**Decoder Training**: Dual-loss backtranslation
- Reconstruction loss (aesthetic fidelity)
- Vector-space loss (semantic preservation)

See [Training README](src/Training/README.md) for complete training guide.

## Repository Structure

```
cs627/
├── src/
│   ├── Encoders/              # Modality → Semantic Vector
│   │   ├── text/              # Text encoders
│   │   │   └── semantic_to_vec.py  # Text_to_Vec (BART) - PRODUCTION
│   │   ├── audio/             # Audio encoders
│   │   │   ├── waveform_to_vec.py  # Audio_to_Vec (Whisper) - PRODUCTION
│   │   │   └── tone_to_vec.py      # Tone_to_Vec (WavLM) - PRODUCTION
│   │   ├── image/             # Image encoders
│   │   │   └── visual_to_vec.py    # Image_to_Vec (ViT) - PRODUCTION
│   │   └── README.md          # How to add new encoders
│   ├── Decoders/              # Semantic Vector → Modality
│   │   ├── text/              # Text decoders
│   │   │   └── vec_to_semantic.py  # Vec_to_Text - PRODUCTION
│   │   ├── image/             # Image decoders
│   │   │   └── vec_to_visual.py    # Vec_to_Image (Stable Diffusion) - EXPERIMENTAL
│   │   ├── audio/             # Audio decoders
│   │   │   └── vec_to_waveform.py  # Vec_to_Audio (TTS) - EXPERIMENTAL
│   │   └── README.md          # How to add new decoders
│   ├── Training/              # Training infrastructure
│   │   ├── encoder_trainers.py     # MoCo contrastive learning
│   │   ├── decoder_trainers.py     # Dual-loss training
│   │   ├── train_encoders.py   # Main training script (raw data)
│   │   ├── Data_Wrangling/
│   │   │   └── mosi_dataset.py     # CMU-MOSI dataset loader
│   │   ├── README.md          # Training documentation
│   │   └── QUICKSTART.md      # Quick start guide
│   ├── Inference/             # Inference applications
│   │   ├── Chatbot/           # Seq2seq chatbot in semantic space
│   │   └── Summarization/     # Text summarization
│   └── utils/
│       └── Adapter.py         # MLP adapter module
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests (smoke tests)
│   ├── integration/           # Integration tests (dataloader tests)
│   └── data_pipeline/         # Data pipeline tests
├── scripts/                   # Utility scripts
│   ├── data_wrangling/        # Data extraction and preprocessing
│   │   ├── download_test_videos.py   # Download MOSI videos
│   │   ├── extract_test_segments.py  # Extract audio/frames
│   │   ├── preprocess_clinc_oos.py   # Intent classification data
│   │   └── preprocess_meld.py        # Emotion classification data
│   └── ...                    # Other utilities
├── .github/workflows/         # CI/CD pipelines
│   └── smoke-tests.yml        # Automated smoke tests for PRs
├── litrature/                 # Research papers and notes
│   ├── Paper.txt              # Companion research paper
│   └── previous_work/         # Related work and references
├── OptimalWeights/            # Trained adapter weights
├── data/                      # Dataset storage (not in git)
├── CLAUDE.md                  # Project guide for Claude Code
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive project guide (architecture, setup, workflows)
- **[CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)**: Cloud GPU deployment guide (AWS, GCP, Azure)
- **[src/Encoders/README.md](src/Encoders/README.md)**: How to create and train encoders
- **[src/Decoders/README.md](src/Decoders/README.md)**: How to create and train decoders
- **[src/Training/README.md](src/Training/README.md)**: Training methodology and best practices
- **[src/Training/QUICKSTART.md](src/Training/QUICKSTART.md)**: Quick start for training

## Datasets

**Primary**: [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) - Multimodal Opinion Sentiment Intensity
- 2,199 opinion video segments
- Modalities: Text transcripts, audio, video
- Accessed via [CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)

**Evaluation**: MultiBench, Conceptual 12M (see [CLAUDE.md](CLAUDE.md) for details)

## Key Features

✅ **Modular architecture**: Encoders, inference, and decoders are independent
✅ **100x training efficiency**: Only train small MLP adapters, not full models
✅ **Cross-modal alignment**: Text ↔ Audio ↔ Video share semantic space
✅ **Momentum contrastive learning**: Stable training with InfoNCE loss
✅ **Backtranslation**: Synthesize training data for unsupervised alignment
✅ **Extensible**: Easy to add new modalities (depth, radar, etc.)
✅ **Automated testing**: Smoke tests and CI/CD via GitHub Actions
✅ **Lazy loading**: Efficient memory usage for large video datasets

## Testing and Quality Assurance

### Running Tests

```bash
# Run smoke tests (unit tests for production encoders)
cd tests/unit
python -m pytest smoke_test.py -v

# Run integration tests (dataloader tests)
cd tests/integration
python test_dataloader_5videos.py
```

### Continuous Integration

All pull requests to `main` must pass automated smoke tests before merge:
- **Encoder initialization tests**: Verify all production encoders load correctly
- **Output shape validation**: Ensure all encoders output (1, 1024) tensors
- **Adapter verification**: Check adapters exist and load properly

See `.github/workflows/smoke-tests.yml` for CI configuration.

## Research

This project builds on:
- **Cross-Modal Momentum Contrastive Learning** [IEEE 2024]
- **MoCo** (Momentum Contrast) [CVPR 2020]
- **BERT-style Adapter Architecture** (APE) [Apple, Paper 11]
- **Meta's Large Concept Model** (LCM) - conceptual inspiration
- **SONAR** - conceptual inspiration for multimodal semantic spaces

See `litrature/` directory for full paper collection and `litrature/Paper.txt` for the companion research paper.

## Contributing

This is a research project for CS627. For questions or contributions, please refer to the companion paper in `litrature/Paper.txt` or contact the team.

## References

- [CMU-MOSI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
- [CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)

---

**Project Status**: Active development | CS627 AI Research
**License**: Research/Educational Use
