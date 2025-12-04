# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CS627 AI research project examining the impact of a shared **Semantic-Vector Space (SVS)** on Natural Language Understanding (NLU) tasks. The project builds on Meta's Large Concept Model work [7] to provide a clear methodology for training and evaluating models that transition data into and out of a Semantic-Vector modality using HuggingFace transformers and adapters.

**Core Contribution**: An Encoder and Decoder evaluation and alignment pipeline that allows for iterative adoption of new input and output modalities to a common Semantic-Vector space.

**Key Insight**: By aligning inference resources to a common Semantic-Vector modality, we create clear separations of concern between Inference, Encoding, and Decoding modules. This allows training and fine-tuning modules in isolation, simplifying training and enabling scientifically rigorous evaluations.

## Setup and Dependencies

### Initial Setup
```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `transformers` (HuggingFace models - BART, Whisper, ViT, Stable Diffusion)
- `torch` (PyTorch deep learning framework)
- `diffusers` (Stable Diffusion models)
- `mteb` and `sentence-transformers` (embedding evaluation)
- `datasets`, `evaluate`, `accelerate`, `trl` (training and evaluation)
- `librosa` and `soundfile` (audio processing)

## Architecture Overview

### Fundamental Concepts

**Semantic-Vector Space**: Instead of repeatedly translating between modalities (which causes semantic drift), this architecture trains all inference models to operate in a shared, dense Semantic-Vector space. Input data is encoded once to this space, inference happens in the vector space, and output is decoded once to the target modality.

**Backtranslation**: Used for data augmentation. Given known pairings (Image, Text) and (Text, Semantic-Vector), we synthesize training targets of (Image, Semantic-Vector). Quality depends on semantic fidelity of the intermediate mappings.

**Contrastive Learning**: Semi-supervised learning using triplets (Query, Positive Sample, Negative Sample) to group similar embeddings together and push dissimilar embeddings apart.

### Core Components

#### 1. Encoders (`src/Encoders/`)
Convert modalities (text, audio, image) into semantic vectors.

**Architecture** (inspired by [11]): Pretrained Encoder + MLP Adapter (Fig. 2 in paper)
- Pretrained module handles modality-specific features
- MLP Adapter (simple multi-layer perceptron) translates to Semantic-Vector space
- This design is highly performant, less prone to overfitting, 2 orders of magnitude cheaper to train, and more robust to distribution shift

**Directory Structure** (organized by modality):
- `text/` - Text encoders (semantic features)
  - `semantic_to_vec.py` - Text encoder (`Text_to_Vec`) using `facebook/bart-base` + Adapter - PRODUCTION
- `audio/` - Audio encoders (waveform, tone, etc.)
  - `waveform_to_vec.py` - Audio encoder (`Audio_to_Vec`) using `openai/whisper-small` + Adapter - PRODUCTION
- `image/` - Image encoders (visual features)
  - `visual_to_vec.py` - Image encoder (`Image_to_Vec`) using `nlpconnect/vit-gpt2-image-captioning` + Adapter - PRODUCTION
- `video/` - Video encoders (motion, temporal features)
- `encoder_boilerplate.py` - Template for creating new encoders

**Imports** (backward compatible):
```python
from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
```

**Current Models**:
- BART (facebook/bart-base) - Text encoding and decoding
- Whisper (openai/whisper-small) - Audio encoding
- ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning) - Image encoding
- Stable Diffusion (CompVis/stable-diffusion-v1-4) - Image decoding (experimental)
- Bark TTS (suno/bark-small) - Audio decoding (experimental)

#### 2. Decoders (`src/Decoders/`)
Convert semantic vectors back to modalities.

**Architecture**: MLP Adapter + Pretrained Decoder
- MLP Adapter translates from Semantic-Vector space
- Pretrained decoder generates modality-specific output

**Important**: To train a Decoder, an Encoder of the same modality must already exist.

**Directory Structure** (organized by modality):
- `text/` - Text decoders (semantic to text)
  - `vec_to_semantic.py` - Vector to text decoder (`Vec_to_Text`) using Adapter + `facebook/bart-base` - PRODUCTION
- `audio/` - Audio decoders (vector to waveform, etc.)
  - `vec_to_waveform.py` - Vector to audio decoder (`Vec_to_Audio`) using `suno/bark-small` - EXPERIMENTAL
- `image/` - Image decoders (vector to visual)
  - `vec_to_visual.py` - Vector to image decoder (`Vec_to_Image`) using `CompVis/stable-diffusion-v1-4` - EXPERIMENTAL
- `video/` - Video decoders (vector to motion, etc.)
- `decoder_boilerplate.py` - Template for creating new decoders

**Imports** (backward compatible):
```python
from Decoders import Vec_to_Text, Vec_to_Audio, Vec_to_Image
```

#### 3. Adapter Module (`src/utils/Adapter.py`)
Neural network bridge between pretrained models and shared semantic space.

**Architecture**: Configurable MLP
- Default: 2 hidden layers, 200 hidden units
- Input/output dimensions: 1024 (configurable)
- ReLU activations between layers

**Usage**:
```python
adapter = Adapter(prefix="model_name", input_length=1024,
                 output_length=1024, hidden_size=200, hidden_layers=2)
adapter.save()  # Saves to OptimalWeights/{prefix}_weights.pth
adapter.load()  # Loads from OptimalWeights/{prefix}_weights.pth
```

### Training Architecture

#### Encoder Alignment (`src/Training/encoder_trainers.py`)
Uses **Cross-Modal Momentum Contrastive Learning** (based on [1]).

**Contrast Trainer**:
- Loss function: Triplet loss with cosine similarity
- Formula: `max(cos_sim(query, positive) - cos_sim(query, negative) + margin, 0)`
- Default margin: 0.1
- Uses cross-modality paired data to align encoders to shared vector space (Fig. 3 in paper)
- Momentum-based distillation overcomes imperfect cross-modality labeling

**Input Format**: `[(query, positive, negative), ...]`

#### Raw Encoder Training (`src/Training/train_encoders.py`)
Training pipeline for encoders using raw multimodal data from CMU-MOSI:

- **Dataset**: `MOSIRawVideoDataset` with lazy loading
- **Collate Function**: `collate_fn_raw_video()` loads audio waveforms (via librosa) and video frames (via PIL) just-in-time
- **Encoders**: Text_to_Vec, Audio_to_Vec, Image_to_Vec
- **Training**: Contrastive learning with cross-modal triplets
- **Memory Optimization**: Lazy loading prevents RAM overflow when training on large video datasets

#### Decoder Alignment
Dual loss approach (Fig. 4 in paper):

1. **Reconstruction Loss**: Measures decoder's fidelity to training data's aesthetic features
2. **Vector-Space Loss**: Measures semantic drift after output is re-encoded to shared vector space
   - Uses negative cosine similarity: `-cos_sim(vecA, vecB)`

Both losses measure reconstruction quality from different perspectives.

**Total Decoder Loss**: `α × Reconstruction Loss + β × Semantic Fidelity Loss`

### Inference Modules (`src/Inference/`)

**Chatbot** (`Chatbot/`):
- `encoder.py` - Bidirectional GRU encoder with word embeddings
- `decoder.py` - Luong attention decoder with greedy search
- `attention.py` - Attention mechanism
- Demonstrates seq2seq inference in semantic space

**Summarization** (`Summarization/`):
- Text summarization tasks using BERT+BART fusion approach

## Data and Preprocessing

### Datasets

**CMU-MOSI** (Primary Dataset): Multimodal Opinion Sentiment Intensity
- 2,199 opinion video segments from YouTube
- Modalities: Text transcripts, Audio waveforms (16kHz), Video frames (RGB)
- Labels: Sentiment intensity scores
- Dataset class: `MOSIRawVideoDataset` in `src/Training/Data_Wrangling/mosi_dataset.py`
- Download function: `download_mosi()` downloads metadata via CMU-MultimodalSDK
- **Data Pipeline**:
  1. Download MOSI metadata: `download_mosi('data/cmumosi/mosi/')`
  2. Extract videos, audio, and frames: `scripts/data_wrangling/extract_test_segments.py`
  3. Load with dataset: `MOSIRawVideoDataset(split='train', mosi_data_path='...', audio_dir='...', video_dir='...')`
- Returns: `{'text': str, 'audio': path, 'video': path, 'label': float, 'segment_id': str}`
- Lazy loading: Audio/video loaded in collate function to avoid RAM overflow

**MultiBench**: Comprehensive evaluation suite with 20 multimodal ML algorithms
- Available: https://github.com/pliang279/MultiBench
- Covers (1) data preprocessing, (2) fusion paradigms, (3) optimization objectives, (4) training procedures

**Conceptual 12M**: ~12 million image-text pairs for vision-and-language pre-training

**Intent Classification** (preprocessed via `scripts/data_wrangling/preprocess_clinc_oos.py`):
- Dataset: clinc_oos (intent classification)
- Returns TF-IDF vectorized train/val/test splits
- Function: `load_and_preprocess_data()` returns `(X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, vectorizer)`

**Emotion Classification** (preprocessed via `scripts/data_wrangling/preprocess_meld.py`):
- Dataset: MELD (emotion classification)
- Returns TF-IDF vectorized train/val/test splits with label encoding
- Function: `load_and_preprocess_data()` returns `(X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, vectorizer, label_encoder)`

## Development Workflow

### Training Notebooks
- `src/encoder_training.ipynb` - Encoder fine-tuning experiments
- `src/encoder_alignment.ipynb` - Encoder alignment to semantic space using contrastive learning
- `src/sonar_sample.ipynb` - SONAR usage examples

### Device Configuration
All models automatically detect GPU availability:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Adapter Weights Management
Adapters use prefix-based naming in `OptimalWeights/`:
```python
self.weights_path = f"OptimalWeights/{prefix}_weights.pth"
```

## Evaluation Metrics

**Primary Metrics** (from paper):
1. Contrastive learning error on unseen data
2. Verification that semantically-linked multi-modal data encodes to similar vectors
3. Cross-modal alignment quality (Text ↔ Image ↔ Audio)

**Performance Benchmarks**:
- Task completion and performance (F1 score, accuracy)
- Inference latency
- Model uncertainty
- Token utilization / inference cost

## Key Design Patterns

1. **BERT Architecture for Coders**: Pretrained Encoder/Decoder + MLP Adapter
   - Efficiency gain: Fine-tuning focuses on small MLP instead of large pretrained model
   - Result: 100x cheaper training, better generalization

2. **Modality Separation**: Clear boundaries between Encoding, Inference, and Decoding
   - Enables independent module training and evaluation
   - Allows mixing and matching different modality combinations

3. **Ground Truth Strategy**: `facebook/bart-base` text encoder serves as the initial ground truth
   - All other encoders align to BART's encoding of text
   - Text modality serves as the bridge between modalities via backtranslation

## Important Notes

- **Base model**: `facebook/bart-base` (BART encoder is the ground truth for semantic space)
- **Audio model**: `openai/whisper-small`
- **Image model**: `nlpconnect/vit-gpt2-image-captioning`
- **Tokenization**: BartTokenizer for text (max_length=1024), WhisperProcessor for audio
- **This is cross-modal work**: Covers text, audio, and image modalities
- **Directory organization**: Encoders and decoders organized by source/target modality (text/, audio/, image/, video/)
- **File naming convention**: `{feature}_to_vec.py` for encoders, `vec_to_{feature}.py` for decoders
- **Class naming convention**: `{Modality}_to_Vec` for encoders, `Vec_to_{Modality}` for decoders
- **Imports**: Use `from Encoders import Text_to_Vec` (backward compatible)
- **All components inherit from**: `torch.nn.Module` (not Pipeline or other base classes)
- **Multiple encoders per modality**: Each modality directory can contain multiple feature-specific encoders (e.g., audio/waveform_to_vec.py, audio/tone_to_vec.py)
- **EXPERIMENTAL status**: New encoders and decoders should be marked EXPERIMENTAL until validated

## Project Documentation

- **ARCHITECTURE.md** - Comprehensive architecture patterns and design principles
- **EVALUATION.md** - Evaluation metrics and testing guidelines
- **CONTRIBUTING.md** - Contribution guidelines for developers
- **src/Encoders/encoder_boilerplate.py** - Template for creating new encoders
- **src/Decoders/decoder_boilerplate.py** - Template for creating new decoders

## Developer Tools

- **tools/create_encoder.py** - Generator script for new encoders (marks as EXPERIMENTAL by default)
- **tools/create_decoder.py** - Generator script for new decoders (marks as EXPERIMENTAL by default)
- **tools/validate_module.py** - Functional validation for production modules (skips experimental)

## Project Structure

### Directory Organization

```
cs627/
├── src/                          # Source code
│   ├── Encoders/                 # Modality to vector encoders
│   │   ├── text/                 # Text encoders (semantic features)
│   │   │   └── semantic_to_vec.py  # Text_to_Vec (PRODUCTION)
│   │   ├── audio/                # Audio encoders (waveform, tone, etc.)
│   │   │   └── waveform_to_vec.py  # Audio_to_Vec (PRODUCTION)
│   │   ├── image/                # Image encoders (visual features)
│   │   │   └── visual_to_vec.py    # Image_to_Vec (PRODUCTION)
│   │   └── video/                # Video encoders (motion, temporal)
│   ├── Decoders/                 # Vector to modality decoders
│   │   ├── text/                 # Text decoders
│   │   │   └── vec_to_semantic.py  # Vec_to_Text (PRODUCTION)
│   │   ├── audio/                # Audio decoders
│   │   │   └── vec_to_waveform.py  # Vec_to_Audio (EXPERIMENTAL)
│   │   ├── image/                # Image decoders
│   │   │   └── vec_to_visual.py    # Vec_to_Image (EXPERIMENTAL)
│   │   └── video/                # Video decoders
│   ├── Training/                 # Training scripts and data loaders
│   │   ├── train_encoders.py   # Main training script
│   │   └── Data_Wrangling/       # Dataset loaders (MOSI, etc.)
│   ├── Inference/                # Inference modules (chatbot, summarization)
│   └── utils/                    # Shared utilities (Adapter, etc.)
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests (smoke_test.py)
│   ├── integration/              # Integration tests (dataloader tests)
│   └── data_pipeline/            # Data pipeline tests
├── scripts/                      # Utility scripts
│   └── data_wrangling/           # Data extraction and preprocessing
├── .github/workflows/            # CI/CD pipelines
│   └── smoke-tests.yml           # Automated tests for PRs
├── OptimalWeights/               # Trained adapter weights
└── data/                         # Dataset storage
```

### Testing and CI/CD

**Smoke Tests** (`tests/unit/smoke_test.py`):
- Quick sanity checks for production-ready encoders
- Tests initialization, input format validation, output shape validation
- Validates all encoders output consistent (1, 1024) shape

**GitHub Actions** (`.github/workflows/smoke-tests.yml`):
- Triggers on pull requests to main/master branches
- Must pass before PR can be merged
- Jobs:
  1. `smoke-test`: Runs unit tests on production encoders
  2. `verify-production-encoders`: Verifies encoder imports and adapter existence

**Running Tests Locally**:
```bash
# Run smoke tests
cd tests/unit
python -m pytest smoke_test.py -v

# Run integration tests
cd tests/integration
python test_dataloader_5videos.py
```

## References

The companion paper is located at `litrature/Paper.txt` and provides detailed methodology and theoretical foundations. Code is publicly available at: https://github.com/WatsonWBlair/cs627
