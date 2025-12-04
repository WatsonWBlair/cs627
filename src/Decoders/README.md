# Decoders

This directory holds models that decode Semantic-Vectors from the shared semantic space into various output modalities (text, audio, image).

## Architecture

All decoders follow the **inverse BERT-style architecture** outlined in the project paper:

```
Semantic-Vector Space → MLP Adapter → Pretrained Decoder → Output Modality
```

**Key Components**:
1. **MLP Adapter** (`utils.Adapter`): Translates semantic vectors to the format expected by the pretrained decoder
2. **Pretrained Decoder**: Modality-specific generative model (BART, Stable Diffusion, TTS, etc.) that produces the output

**Important**: To train a Decoder for a given modality, an **Encoder of the same modality must already exist**. This is required for the backtranslation training process.

## Current Decoders

### Text Decoder (`vec_2_text.py`)
- **Class**: `Vec_to_Text`
- **Base Model**: `facebook/bart-base`
- **Tokenizer**: `BartTokenizer`
- **Purpose**: Converts semantic vectors back to natural language text
- **Architecture**: Semantic Vector → Adapter → BART decoder → Text
- **Status**: Fully implemented

### Image Decoder (`vec_2_img.py`)
- **Class**: `Vec_to_Image`
- **Base Model**: `CompVis/stable-diffusion-v1-4`
- **Pipeline**: `StableDiffusionPipeline`
- **Purpose**: Generates images from semantic vectors
- **Architecture**: Semantic Vector → Adapter (replaces text encoder) → Stable Diffusion → Image
- **Status**: EXPERIMENTAL - Adapter integration with Stable Diffusion needs validation

### Audio Decoder (`vec_2_audio.py`)
- **Class**: `Vec_to_Audio`
- **Base Model**: `suno/bark-small` (text-to-speech)
- **Purpose**: Generates speech audio from semantic vectors
- **Architecture**: Semantic Vector → Adapter → Text (intermediate) → TTS Pipeline → Audio
- **Status**: EXPERIMENTAL - Semantic vector to text conversion not yet implemented

## How to Add a New Decoder

### Step 1: Verify Encoder Exists

**REQUIRED**: Before creating a decoder, ensure an encoder for the same modality exists in `src/Encoders/`.

This encoder will be used for:
- Generating training data via backtranslation
- Computing reconstruction loss during training
- Validating semantic vector fidelity

### Step 2: Create the Decoder Class

Use the boilerplate in `decoder_boilerplate.py` as your template. Key structure:

```python
class Vec_to_{Modality}(torch.nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, output_dim: int = 1024) -> None:
        self.adapter: Adapter = Adapter(...)
        self.decoder: DecoderModel = DecoderModel.from_pretrained(base_model)
```

See `decoder_boilerplate.py` for full implementation template.

### Alternative: Pipeline-Based Decoders

For generative pipelines (like Stable Diffusion), replace components with the Adapter:

```python
import torch
from diffusers import YourPipeline
from utils.Adapter import Adapter

BASE_MODEL = "your/pipeline-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Vec_to_YourModality:
    def __init__(self, base_model: str = BASE_MODEL, token_length: int = 1024):
        super(Vec_to_YourModality, self).__init__()

        # 1. Load the generative pipeline
        self.pipeline = YourPipeline.from_pretrained(base_model)

        # 2. Replace the text/prompt encoder with Adapter
        self.pipeline.text_encoder = Adapter(
            prefix=f"{base_model}_dec",
            input_length=token_length,
            output_length=token_length,
            hidden_size=200,
            hidden_layers=2
        )

    def forward(self, semantic_vector, device: str = DEVICE):
        # Generate output directly from semantic vector
        output = self.pipeline.to(device)(prompt=semantic_vector).images[0]
        return output
```

### Step 3: Train the Decoder

Use the **dual-loss trainer** in `src/Training/decoder_trainers.py`:

1. **Prepare Training Data**:
   ```python
   # For each training sample:
   # 1. Encode original data to semantic vector (using corresponding encoder)
   # 2. Decode semantic vector back to modality (using your decoder)
   # 3. Re-encode decoded output (using corresponding encoder)
   ```

2. **Use the Contrast Trainer**:
   ```python
   from Training.decoder_trainers import Contrast

   trainer = Contrast(
       model=your_decoder,
       # ... other Trainer arguments
   )
   trainer.train()
   ```

3. **Dual Loss Calculation**:
   - **Reconstruction Loss**: Measures aesthetic/perceptual fidelity
     - Compares original data vs. decoded output
     - Use modality-specific metrics (SSIM for images, DICE, etc.)
   - **Vector-Space Loss**: Measures semantic drift
     - Formula: `-cos_sim(original_vector, re_encoded_vector)`
     - Ensures semantic meaning is preserved through decode→encode cycle

4. **Save Adapter Weights**:
   ```python
   your_decoder.adapter.save()  # Saves to OptimalWeights/{prefix}_weights.pth
   ```

### Step 4: Validate Backtranslation

Test the full encode→decode→encode cycle:

```python
# 1. Encode original data
original_vector = encoder(original_data)

# 2. Decode to modality
reconstructed_data = decoder(original_vector)

# 3. Re-encode reconstructed data
reconstructed_vector = encoder(reconstructed_data)

# 4. Measure semantic drift
import torch.nn.functional as F
semantic_similarity = F.cosine_similarity(original_vector, reconstructed_vector)
print(f"Semantic Fidelity: {semantic_similarity.item():.4f}")  # Should be high (>0.8)
```

### Step 5: Evaluate Reconstruction Quality

For each modality, use appropriate quality metrics:

- **Text**: BLEU, ROUGE, perplexity
- **Images**: SSIM, PSNR, FID (Fréchet Inception Distance)
- **Audio**: MOS (Mean Opinion Score), spectral analysis

### Step 6: Document in Notebooks

Add training and evaluation examples to:
- `src/encoder_training.ipynb` - Combined encoder-decoder training
- `src/encoder_alignment.ipynb` - Cross-modal alignment validation

## Training Strategy: Backtranslation

Decoders use **backtranslation** for training data synthesis:

1. **Given Pairings**:
   - (Original Data, Semantic Vector) from trained encoder

2. **Training Process**:
   - Input: Semantic vector
   - Output: Reconstructed data
   - Loss: Combination of reconstruction loss + vector-space loss

3. **Quality Factors**:
   - Encoder quality directly impacts decoder training quality
   - Better aligned encoders → better decoder training data
   - Iterative improvement: Train encoder → train decoder → refine encoder → refine decoder

## Loss Functions (from `decoder_trainers.py`)

### Reconstruction Loss
```python
def reconstruction_loss(self, original, reconstructed):
    # Modality-specific comparison
    # Examples: SSIM for images, DICE for segmentation, MSE for audio
    pass
```

### Vector-Space Loss
```python
def vector_loss(self, vecA, vecB):
    # Measures semantic drift via cosine similarity
    maxMe = F.cosine_similarity(vecA, vecB)
    return -maxMe  # Negative because we want to maximize similarity
```

## Important Notes

1. **Encoder Dependency**: You MUST have a working encoder before training a decoder
2. **Quality Chain**: Decoder quality ≤ Encoder quality (garbage in, garbage out)
3. **Semantic Fidelity**: Prioritize vector-space loss if semantic meaning matters more than aesthetic quality
4. **Device Support**: Always include CUDA/CPU detection for portability
5. **Output Format**: Ensure your decoder returns data in standard formats (text strings, PIL Images, WAV arrays, etc.)

## Target Decoder Models (from paper)

The paper proposes these decoders for evaluation:
- **MiniGPT** - Text generation
- **Stable Diffusion** - Image generation
- **TTS Models** - Speech synthesis

## Archive

The `archive/` directory contains:
- **Old SEED implementations**: Previous decoder experiments
- **Legacy decoders**: `text/archive/` contains sentence-level VAE implementations
- **Experimental code**: Historical decoder approaches

**Do not modify archive code** unless explicitly working with legacy implementations.

## Common Pitfalls

1. **Dimension Mismatch**: Ensure adapter output dimension matches decoder input dimension
2. **Missing Encoder**: Cannot train decoder without corresponding encoder
3. **Poor Backtranslation**: Low-quality encoder produces poor decoder training data
4. **Loss Imbalance**: Balance reconstruction loss and vector-space loss (both are important)
5. **Pipeline Integration**: When replacing pipeline components (like `text_encoder`), verify compatibility

