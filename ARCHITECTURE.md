# Architecture Patterns

**CS627 Semantic-Vector Space Project**

This document defines the standard architectural patterns and conventions for all code in this project.

---

## Core Design Principle

This project follows a **shared semantic-vector space** architecture where all modalities are encoded into a common 1024-dimensional representation space.

```
Text ──┐
Audio ─┼──> Encoders ──> Semantic Vector Space (1024-dim) ──> Decoders ──┬──> Text
Image ─┘                                                                   ├──> Audio
...                                                                        └──> Image
```

**Key Benefits**:
- Cross-modal understanding (text ↔ audio ↔ image)
- Modular training (train components independently)
- Efficient inference (no modality-specific pipelines)
- Scientific rigor (clear separation of encoding, inference, decoding)

---

## Standard Patterns

### Encoder Pattern

**Purpose**: Convert input modality → semantic vector (1024-dim)

**Architecture Flow**:
```
Input Modality → Pretrained Encoder → MLP Adapter → Semantic Vector (1024-dim)
```

**Key Characteristics**:
- Pretrained encoder is frozen (prevents catastrophic forgetting)
- Only MLP adapter is trained (100x cheaper than full fine-tuning)
- Adapter maps encoder output (varies by model) → 1024-dim semantic space
- Uses mean pooling over sequence dimension

**Implementation**:
```python
# See: src/Encoders/encoder_boilerplate.py for full template
encoder = Text_to_Vec(base_model="facebook/bart-base", output_dim=1024)
semantic_vector = encoder(input_text)  # Returns (batch, 1024)
```

**Examples**: `text_2_vec.py`, `wav_2_vec.py`, `img_2_vec.py`

---

### Decoder Pattern

**Purpose**: Convert semantic vector (1024-dim) → output modality

**Architecture Flow**:
```
Semantic Vector (1024-dim) → MLP Adapter → Pretrained Decoder → Output Modality
```

**Key Characteristics**:
- Adapter translates from semantic space to decoder input format
- Requires corresponding encoder for backtranslation training
- May be experimental (marked with EXPERIMENTAL tag)

**Implementation**:
```python
# See: src/Decoders/decoder_boilerplate.py for full template
decoder = Vec_to_Text(base_model="facebook/bart-base", output_dim=1024)
text_output = decoder(semantic_vector)  # Returns str or List[str]
```

**Examples**: `vec_2_text.py`, `vec_2_audio.py` (experimental), `vec_2_img.py` (experimental)

---

### Adapter Module

**Purpose**: Small MLP bridge between encoders/decoders and semantic space

**Architecture**: `Input → Linear → ReLU → [Hidden Layers] → Linear → Output`

**Usage**:
```python
# For encoder: modality → semantic space
adapter = Adapter(prefix="model_enc", input_length=768, output_length=1024)

# For decoder: semantic space → modality
adapter = Adapter(prefix="model_dec", input_length=1024, output_length=768)

adapter.save()  # Saves to AdapterWeights/{prefix}_weights.pth
```

**Parameters**:
- `prefix`: Unique identifier (determines weight file name)
- `input_length`: Input dimension
- `output_length`: Output dimension
- `hidden_size`: Hidden layer width (default: 200)
- `hidden_layers`: Number of hidden layers (default: 2)

---

## Naming Conventions

### Files
| Component | Pattern | Example |
|-----------|---------|---------|
| Encoder | `{modality}_2_vec.py` | `text_2_vec.py` |
| Decoder | `vec_2_{modality}.py` | `vec_2_text.py` |

**Rule**: Lowercase with underscores, use `2` (not `to`)

### Classes
| Component | Pattern | Example |
|-----------|---------|---------|
| Encoder | `{Modality}_to_Vec` | `Text_to_Vec` |
| Decoder | `Vec_to_{Modality}` | `Vec_to_Text` |

**Rule**: PascalCase with underscores, use `_to_` (not `_2_`)

### Methods
- `__init__(self, base_model: str, ...) -> None`
- `forward(self, input, *, device: str = DEVICE) -> torch.Tensor`

**Rule**: Use keyword-only args after positional (`*,` separator)

---

## Type Annotations

**Required for core components**:
```python
def __init__(self, base_model: str = BASE_MODEL, output_dim: int = 1024) -> None:
    ...

def forward(self, input: Union[str, List[str]], *, device: str = DEVICE) -> torch.Tensor:
    ...
```

**Import**: `from typing import Union, Optional, List`

---

## Training Patterns

### Encoder Training: Momentum Contrastive Learning (MoCo)

**Approach**: Cross-modal alignment using InfoNCE loss

```python
from Training.encoder_trainers import Contrast

encoders = {'text': Text_to_Vec(), 'audio': Audio_to_Vec()}
trainer = Contrast(model=encoders, momentum=0.999, queue_size=65536, temperature=0.07)
trainer.train()
```

**Key Hyperparameters**:
- Momentum: 0.999 (EMA update rate)
- Queue size: Start 4096, scale to 65536
- Temperature: 0.07 (InfoNCE scaling)

### Decoder Training: Dual-Loss Backtranslation

**Approach**: Reconstruction + semantic preservation

```python
# Encode → Decode → Re-encode
original_vec = encoder(input_data)
reconstructed = decoder(original_vec)
reconstructed_vec = encoder(reconstructed)

# Dual loss
loss = α * reconstruction_loss(input_data, reconstructed) + \
       β * (-cosine_similarity(original_vec, reconstructed_vec))
```

**Reconstruction Loss**: Modality-specific (BLEU for text, SSIM for images, etc.)

---

## Device Handling

**Pattern**: Keyword-only device parameter with automatic placement

```python
def forward(self, input, *, device: str = DEVICE):
    self.model = self.model.to(device)
    input = input.to(device)
    return self.model(input)
```

---

## Directory Structure

```
cs627/
├── src/
│   ├── Encoders/
│   │   ├── encoder_boilerplate.py  # Template for new encoders
│   │   ├── text_2_vec.py
│   │   ├── wav_2_vec.py
│   │   └── img_2_vec.py
│   ├── Decoders/
│   │   ├── decoder_boilerplate.py  # Template for new decoders
│   │   ├── vec_2_text.py
│   │   ├── vec_2_audio.py
│   │   └── vec_2_img.py
│   ├── Training/
│   │   └── encoder_trainers.py     # MoCo implementation
│   └── utils/
│       └── Adapter.py
├── AdapterWeights/                 # Saved weights ({prefix}_weights.pth)
├── tools/                          # Generator scripts
│   ├── create_encoder.py
│   ├── create_decoder.py
│   └── validate_module.py
└── ARCHITECTURE.md                 # This file
```

---

## Creating New Modules

### Option 1: Use Generator Scripts (Recommended)

```bash
python tools/create_encoder.py --modality video --model facebook/timesformer-base
python tools/create_decoder.py --modality video --model facebook/timesformer-base
```

### Option 2: Copy Boilerplate Manually

```bash
# For encoder
cp src/Encoders/encoder_boilerplate.py src/Encoders/video_2_vec.py
# Edit: Update class name, model, processor, dimensions

# For decoder
cp src/Decoders/decoder_boilerplate.py src/Decoders/vec_2_video.py
# Edit: Update class name, model, processor, output type
```

---

## Validation Checklist

Before committing:

**Architecture**:
- [ ] Inherits from `torch.nn.Module`
- [ ] Class naming follows convention
- [ ] Adapter initialized with `prefix`, `input_length`, `output_length`

**Type Safety**:
- [ ] Parameters have type hints
- [ ] Return types specified

**Testing**:
```bash
python tools/validate_module.py src/Encoders/your_encoder.py
python -m py_compile src/Encoders/your_encoder.py
python -c "from Encoders.your_encoder import Your_Encoder"
```

---

## Common Gotchas

### ❌ Wrong Adapter Initialization
```python
adapter = Adapter(token_length=1024)  # Missing prefix!
```

### ✅ Correct
```python
adapter = Adapter(prefix="model_enc", input_length=768, output_length=1024)
```

### ❌ Wrong Base Class
```python
class Vec_to_Text(Pipeline):  # Pipeline is inference-only!
```

### ✅ Correct
```python
class Vec_to_Text(torch.nn.Module):
```

### ❌ Wrong Device Handling
```python
def forward(self, input, device="cuda"):  # Positional arg!
```

### ✅ Correct
```python
def forward(self, input, *, device: str = DEVICE):  # Keyword-only!
```

---

## Resources

- **Boilerplates**: `src/Encoders/encoder_boilerplate.py`, `src/Decoders/decoder_boilerplate.py`
- **Training Guide**: [src/Training/README.md](src/Training/README.md)
- **Evaluation Guide**: [EVALUATION.md](EVALUATION.md)
- **Developer Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Project Overview**: [README.md](README.md)

---

**Version**: 1.0
**Last Updated**: 2025-12-01
