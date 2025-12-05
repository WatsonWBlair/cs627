# Encoders

Models that encode modalities (text, audio, image) into the shared Semantic-Vector space.

## Architecture

```
Input Modality → Pretrained Encoder (frozen) → MLP Adapter → Semantic Vector (1024-dim)
```

This design is **100x cheaper to train** - only the small MLP adapter is trained.

## Current Encoders

| Class | File | Model | Purpose |
|-------|------|-------|---------|
| `Text_to_Vec` | `text/semantic_to_vec.py` | facebook/bart-base | Ground truth encoder |
| `Audio_to_Vec` | `audio/waveform_to_vec.py` | openai/whisper-small | Speech to semantic |
| `Tone_to_Vec` | `audio/tone_to_vec.py` | microsoft/wavlm-base | Prosody and emotion |
| `Image_to_Vec` | `image/visual_to_vec.py` | nlpconnect/vit-gpt2-image-captioning | Visual to semantic |

## Adding New Encoders

1. Use `encoder_boilerplate.py` as template
2. Or run: `python tools/create_encoder.py --modality <name> --model <hf_model>`
3. Train using contrastive learning - see [Training README](../Training/README.md)

## Important Notes

- All encoders output shape: `(batch_size, 1024)`
- Ground truth: BART text encoder defines the semantic space
- New encoders align to BART via backtranslation
- Mark new encoders as EXPERIMENTAL until validated
