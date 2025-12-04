# Training

This directory contains trainers and scripts for aligning encoders and decoders to the shared Semantic-Vector space.

## Overview

The training infrastructure implements **Cross-Modal Momentum Contrastive Learning (MoCo)** to align multimodal encoders (text, audio, video) in a shared semantic space. This approach is based on recent research in cross-modal representation learning.

## Quick Start

### 1. Install Dependencies

```bash
# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install .

# Install other dependencies
pip install -r requirements.txt
```

### 2. Run Encoder Alignment Training

```bash
python src/Training/train_encoder_alignment.py
```

This will:
- Download and preprocess CMU-MOSI dataset
- Initialize text, audio, and video encoders
- Train adapters using momentum contrastive learning
- Evaluate cross-modal retrieval performance
- Save trained adapter weights to `OptimalWeights/`

## Architecture

### Cross-Modal Momentum Contrastive Learning (MoCo)

Our training approach uses three key components:

#### 1. Momentum Encoder
A slowly-updated copy of the encoder that provides consistent feature representations:

```
θ_momentum ← m * θ_momentum + (1 - m) * θ_encoder
```

where `m = 0.999` (momentum coefficient)

**Why?** Prevents the encoder from changing too quickly, which would invalidate the memory queue.

#### 2. Memory Queue
A large FIFO queue (size: 4,096 - 65,536) that stores encoded features from previous batches:

```
Queue: [feature_1, feature_2, ..., feature_K]
```

**Why?** Provides a large pool of negative samples without requiring enormous batch sizes.

#### 3. InfoNCE Loss
Temperature-scaled contrastive loss:

```
Loss = -log(exp(q·k+ / τ) / (exp(q·k+ / τ) + Σ exp(q·k- / τ)))
```

where:
- `q` = query embedding (e.g., text)
- `k+` = positive key (e.g., aligned audio from same video)
- `k-` = negative keys (from memory queue)
- `τ = 0.07` (temperature parameter)

**Why?** More stable than triplet margin loss; scales better to large negative pools.

## Files

### Trainers (`encoder_trainers.py`)

#### `Contrast` Class

The main trainer for cross-modal alignment.

**Key Features:**
- Supports both single-encoder and multi-encoder training
- Implements momentum encoder with exponential moving average
- Manages memory queue for large-scale negative sampling
- Computes InfoNCE loss for contrastive learning
- Backward compatible with legacy triplet loss

**Usage:**

```python
from Training.encoder_trainers import Contrast
from Encoders.text_2_vec import Text_to_Vec
from Encoders.wav_2_vec import Audio_to_Vec

# Initialize encoders
encoders = {
    'text': Text_to_Vec(),
    'audio': Audio_to_Vec()
}

# Create trainer
trainer = Contrast(
    model=encoders,
    momentum=0.999,        # Momentum coefficient
    queue_size=65536,      # Memory queue size
    temperature=0.07,      # InfoNCE temperature
    embed_dim=1024,        # Embedding dimension
    use_momentum=True,     # Enable MoCo
    args=training_args
)

# Train
trainer.train()
```

**Parameters:**
- `momentum` (float): Momentum coefficient for updating momentum encoder (default: 0.999)
- `queue_size` (int): Size of negative sample queue (default: 65536)
- `temperature` (float): Temperature for InfoNCE loss (default: 0.07)
- `embed_dim` (int): Dimension of semantic vectors (default: 1024)
- `use_momentum` (bool): Whether to use momentum encoder (default: True)

**Input Formats:**

1. **Multi-modal dict** (recommended):
```python
inputs = {
    'text': text_tensor,
    'audio': audio_tensor,
    'video': video_tensor
}
```

2. **Legacy triplet tuples**:
```python
inputs = [(query, positive, negative), ...]
```

### Data Wrangling (`Data_Wrangling/`)

#### `mosi_dataset.py`

CMU-MOSI dataset loader for cross-modal contrastive learning.

**Functions:**

- `download_mosi(data_path)`: Download CMU-MOSI dataset using CMU-MultimodalSDK
- `preprocess_mosi(data_path)`: Preprocess and split dataset into train/val/test
- `get_mosi_dataloader(split, batch_size)`: Create PyTorch DataLoader

**MOSIDataset Class:**

```python
from Training.Data_Wrangling.mosi_dataset import MOSIDataset

dataset = MOSIDataset(
    data_path='data/cmumosi/',
    split='train',           # 'train', 'valid', or 'test'
    return_labels=False,     # Whether to return sentiment labels
    max_text_length=512
)
```

**Data Format:**

Each sample contains aligned modalities from one video segment:

```python
{
    'text': text_transcript,      # str or tensor
    'audio': audio_features,       # tensor
    'video': video_features,       # tensor
    'segment_id': 'video_id',      # str
    'label': sentiment_score       # float (if return_labels=True)
}
```

**Dataset Statistics:**
- Total segments: ~2,199
- Train: 70% (~1,540 samples)
- Validation: 15% (~330 samples)
- Test: 15% (~330 samples)

#### CMU-MOSI Dataset

**Modalities:**
- **Text**: Transcripts of spoken opinions
- **Audio**: Acoustic features from speech
- **Video**: Visual features (facial expressions, gestures)

**Labels:**
- Sentiment intensity scores (continuous)

**Source:** CMU Multimodal Opinion Sentiment Intensity dataset

### Training Scripts

#### `train_encoder_alignment.py`

Main training script for cross-modal encoder alignment.

**Configuration:**

```python
# Training hyperparameters
BATCH_SIZE = 32           # Adjust based on GPU memory
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500

# MoCo hyperparameters
MOMENTUM = 0.999
QUEUE_SIZE = 4096         # Start smaller, increase if memory allows
TEMPERATURE = 0.07
EMBED_DIM = 1024
```

**Training Pipeline:**

1. **Data Setup**: Download and preprocess CMU-MOSI
2. **Encoder Initialization**: Load text, audio, video encoders
3. **Freeze Pretrained Models**: Only train adapter layers
4. **Initial Evaluation**: Measure cross-modal retrieval before training
5. **Training**: Run MoCo contrastive learning
6. **Final Evaluation**: Measure improvements
7. **Save Weights**: Store adapter weights

**Evaluation Metrics:**

- **Recall@K**: Cross-modal retrieval accuracy
  - `text_to_audio_R@1`, `R@5`, `R@10`
  - `text_to_video_R@1`, `R@5`, `R@10`

- **Average Cosine Similarity**: Semantic alignment quality
  - `avg_text_audio_similarity`
  - `avg_text_video_similarity`

**Success Criteria:**

- Recall@1 > 50%
- Recall@5 > 80%
- Average cosine similarity > 0.8

## Training Best Practices

### 1. Start with Small Queue Size

Begin with `queue_size=4096` and gradually increase:
- 4,096: Good for 8-16GB GPU
- 16,384: Requires 16-32GB GPU
- 65,536: Ideal performance, requires 32GB+ GPU

### 2. Only Train Adapters

Freeze all pretrained model parameters:

```python
# Freeze pretrained encoder
for param in encoder.model.parameters():
    param.requires_grad = False

# Unfreeze adapter
for param in encoder.adapter.parameters():
    param.requires_grad = True
```

**Why?**
- 100x cheaper to train
- Prevents overfitting
- Maintains pretrained knowledge

### 3. Use Mixed Precision Training

Enable FP16 for faster training:

```python
training_args = TrainingArguments(
    fp16=True,  # or bf16=True for Ampere GPUs
    ...
)
```

### 4. Monitor Cross-Modal Alignment

Check validation metrics regularly:
- InfoNCE loss should decrease and stabilize
- Cross-modal similarity should increase
- Retrieval accuracy should improve

### 5. Batch Size Recommendations

| GPU Memory | Batch Size | Queue Size |
|------------|------------|------------|
| 8 GB       | 16-32      | 2,048      |
| 16 GB      | 32-64      | 4,096      |
| 24 GB      | 64-128     | 16,384     |
| 40 GB+     | 128-256    | 65,536     |

## Advanced Usage

### Custom Modality Pairs

Train only specific modality pairs:

```python
# Only align text and audio
encoders = {
    'text': Text_to_Vec(),
    'audio': Audio_to_Vec()
}
```

### Adjust Temperature

Lower temperature = harder negatives:

```python
trainer = Contrast(
    temperature=0.05,  # Stricter similarity threshold
    ...
)
```

### Disable Momentum (Simpler Training)

For debugging or faster experiments:

```python
trainer = Contrast(
    use_momentum=False,  # Simpler contrastive learning
    ...
)
```

## Decoder Training

See `decoder_trainers.py` for decoder-specific training with dual loss (reconstruction + vector-space).

**Key Difference:**
- **Encoders**: Align modalities to shared space (contrastive learning)
- **Decoders**: Generate modality from semantic vectors (reconstruction + alignment)

## References

1. **Cross-Modal Momentum Contrastive Learning**:
   - [Cross-Modal Alignment for End-to-End Spoken Language Understanding](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10448143) (IEEE 2024)

2. **Momentum Contrast (MoCo)**:
   - [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) (CVPR 2020)

3. **InfoNCE Loss**:
   - [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (NeurIPS 2018)

4. **CMU-MOSI Dataset**:
   - [Multimodal Opinion Sentiment Intensity Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce `batch_size`
2. Reduce `queue_size`
3. Enable gradient checkpointing
4. Use FP16 mixed precision

### Low Retrieval Accuracy

**Potential Issues:**
1. Insufficient training epochs
2. Temperature too high (try lowering to 0.05)
3. Learning rate too low/high
4. Data preprocessing issues

### InfoNCE Loss Not Decreasing

**Potential Issues:**
1. Queue size too small (increase to 16,384+)
2. Momentum coefficient too high (try 0.99 instead of 0.999)
3. Encoders not properly initialized
4. Adapter weights not being updated (check `requires_grad`)

## Next Steps

After successful encoder alignment:

1. **Train Decoders**: Use aligned encoders to train decoders (see `src/Decoders/README.md`)
2. **Cross-Modal Inference**: Build applications that leverage aligned semantic space
3. **Fine-tune on Task**: Adapt aligned encoders to specific downstream tasks
4. **Expand Modalities**: Add new encoders (e.g., depth, radar, thermal)

## Output

**Trained Adapter Weights:**
- `OptimalWeights/facebook_bart-base_text_enc_weights.pth`
- `OptimalWeights/openai_whisper-small_audio_enc_weights.pth`
- `OptimalWeights/nlpconnect_vit-gpt2-image-captioning_image_enc_weights.pth`

**Training Logs:**
- `results/encoder_alignment/`

**Checkpoints:**
- `results/encoder_alignment/checkpoint-{step}/`
