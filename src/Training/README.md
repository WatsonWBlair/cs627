# Training

Training infrastructure for cross-modal encoder and decoder alignment.

## Quick Start

```bash
# Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install . && cd ..

# Run encoder training
python src/Training/train_encoders.py

# Run decoder training
TRAIN_TEXT=1 python src/Training/train_decoders.py
```

## Encoder Training

Uses **Cross-Modal Momentum Contrastive Learning (MoCo)**:
- Momentum encoder for stable features
- Memory queue for large negative pools
- InfoNCE loss (temperature-scaled)

### Key Files

| File | Purpose |
|------|---------|
| `train_encoders.py` | Main training script |
| `encoder_trainers.py` | `Contrast` trainer class |
| `Data_Wrangling/mosi_dataset.py` | CMU-MOSI dataset loader |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Batch size |
| `LEARNING_RATE` | 1e-4 | Learning rate |
| `MOMENTUM` | 0.999 | MoCo momentum |
| `QUEUE_SIZE` | 4096 | Negative sample queue |
| `TEMPERATURE` | 0.07 | InfoNCE temperature |

### Success Criteria
- Recall@1 > 50%
- Recall@5 > 80%
- Average cosine similarity > 0.8

## Decoder Training

Uses **dual-loss training**:
- Reconstruction loss (modality-specific)
- Semantic fidelity loss (cosine similarity)

### Key Files

| File | Purpose |
|------|---------|
| `train_decoders.py` | Main training script |
| `decoder_trainer.py` | `CrossModalDecoderTrainer` class |
| `decoder_losses.py` | Modality-specific loss functions |

### Environment Variables

```bash
TRAIN_TEXT=1      # Train text decoder
TRAIN_AUDIO=1     # Train audio decoder
TRAIN_IMAGE=1     # Train image decoder
SEMANTIC_SOURCE=text_only  # or 'matched', 'mixed'
```

## Dataset: CMU-MOSI

- 2,199 video segments with aligned text, audio, video
- Auto-downloaded via CMU-MultimodalSDK
- Split: 70% train, 15% val, 15% test

## Output

Trained weights saved to `OptimalWeights/`:
- `facebook_bart_base_text_enc_weights.pth`
- `openai_whisper-small_audio_enc_weights.pth`
- `nlpconnect_vit-gpt2-image-captioning_image_enc_weights.pth`

## GPU Memory Guidelines

| GPU VRAM | Batch Size | Queue Size |
|----------|------------|------------|
| 8 GB | 16 | 2,048 |
| 16 GB | 32 | 4,096 |
| 24 GB | 64 | 16,384 |

## Troubleshooting

**Out of Memory**: Reduce `BATCH_SIZE` or `QUEUE_SIZE`

**Low Retrieval Accuracy**: Train longer, lower temperature (0.05), increase queue size
