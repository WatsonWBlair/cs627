# Decoders

Models that decode Semantic-Vectors into output modalities (text, audio, image).

## Architecture

```
Semantic Vector (1024-dim) → MLP Adapter → Pretrained Decoder → Output Modality
```

**Prerequisite**: An encoder for the same modality must exist (for backtranslation training).

## Current Decoders

| Class | File | Model | Status |
|-------|------|-------|--------|
| `Vec_to_Text` | `text/vec_to_semantic.py` | facebook/bart-base | PRODUCTION |
| `Vec_to_Audio` | `audio/vec_to_waveform.py` | microsoft/speecht5_tts | EXPERIMENTAL |
| `Vec_to_Image` | `image/vec_to_visual.py` | CompVis/stable-diffusion-v1-4 | EXPERIMENTAL |

## Adding New Decoders

1. Verify encoder exists for the modality
2. Use `decoder_boilerplate.py` as template
3. Or run: `python tools/create_decoder.py --modality <name> --model <hf_model>`
4. Train using dual-loss approach - see [Training README](../Training/README.md)

## Training

Decoders use **dual-loss training**:
- **Reconstruction Loss**: Modality-specific fidelity (CE for text, MSE for audio, CLIP for images)
- **Semantic Fidelity Loss**: Cosine similarity after re-encoding

## Important Notes

- Encoder must exist before training decoder
- Decoder quality depends on encoder quality
- Mark new decoders as EXPERIMENTAL until validated
- Standard output formats: strings (text), numpy arrays (audio), PIL Images (image)
