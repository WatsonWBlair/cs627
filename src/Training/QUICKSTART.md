# Quick Start: Cross-Modal Encoder Alignment

This guide gets you started with training multimodal encoders on CMU-MOSI dataset.

## Prerequisites

```bash
# 1. Install CMU-MultimodalSDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK && pip install .
cd ..

# 2. Install project dependencies
pip install -r requirements.txt
```

## Step 1: Run Training

```bash
python src/Training/train_encoders.py
```

That's it! The script will:
1. ✓ Download CMU-MOSI dataset (~2,199 video segments)
2. ✓ Preprocess and split into train/val/test
3. ✓ Initialize text, audio, video encoders
4. ✓ Train adapters using MoCo contrastive learning
5. ✓ Evaluate cross-modal retrieval performance
6. ✓ Save adapter weights to `OptimalWeights/`

## Step 2: Monitor Training

Watch for these metrics in the console:

```
InfoNCE Loss: 2.45 → 0.85  (should decrease)
text_to_audio_R@1: 0.23 → 0.67  (should increase)
avg_text_audio_similarity: 0.42 → 0.86  (should increase)
```

**Success Criteria:**
- InfoNCE loss < 1.0
- Recall@1 > 50%
- Avg similarity > 0.8

## Step 3: Use Trained Encoders

```python
from src.Encoders.text.semantic_to_vec import Text_to_Vec
from src.Encoders.audio.waveform_to_vec import Audio_to_Vec

# Load encoders (adapters will load saved weights automatically)
text_encoder = Text_to_Vec()
audio_encoder = Audio_to_Vec()

# Encode data
text_vector = text_encoder("This is a test sentence")
audio_vector = audio_encoder(audio_waveform)

# Measure similarity
import torch.nn.functional as F
similarity = F.cosine_similarity(text_vector, audio_vector)
print(f"Cross-modal similarity: {similarity.item():.4f}")
```

## Configuration

Edit `train_encoders.py` to adjust:

```python
# GPU Memory → Batch Size → Queue Size
BATCH_SIZE = 32      # 16-256 depending on GPU
QUEUE_SIZE = 4096    # 2048-65536 depending on memory
TEMPERATURE = 0.07   # Lower = stricter alignment
MOMENTUM = 0.999     # Higher = slower updates
```

## Troubleshooting

### Out of Memory
```python
BATCH_SIZE = 16      # Reduce batch size
QUEUE_SIZE = 2048    # Reduce queue size
```

### Low Performance
```python
NUM_EPOCHS = 200     # Train longer
TEMPERATURE = 0.05   # Stricter alignment
QUEUE_SIZE = 16384   # More negatives
```

### Data Download Issues
```python
# Manual download:
# 1. Visit http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/
# 2. Download dataset manually
# 3. Place in data/cmumosi/
```

## Next Steps

- **Train Decoders**: See `src/Decoders/README.md`
- **Build Applications**: Use aligned encoders for cross-modal tasks
- **Add Modalities**: Extend to new input types
- **Fine-tune**: Adapt to specific downstream tasks

## Output Files

After training:

```
OptimalWeights/
├── facebook_bart-base_text_enc_weights.pth
├── openai_whisper-small_audio_enc_weights.pth
└── nlpconnect_vit-gpt2-image-captioning_image_enc_weights.pth

results/encoder_alignment/
├── checkpoint-1000/
├── checkpoint-2000/
└── training_logs.txt
```

## Resources

- Full documentation: `src/Training/README.md`
- Trainer details: `src/Training/encoder_trainers.py`
- Dataset loader: `src/Training/Data_Wrangling/mosi_dataset.py`
- Paper: `literature/previous_work/Cross-Modal_Alignment_for_End-to-End_Spoken_Language_Understanding_Based_on_Momentum_Contrastive_Learning.pdf`
