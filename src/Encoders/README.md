# Encoders

This directory holds models that encode various modalities (text, audio, image) into the shared Semantic-Vector space.

## Architecture

All encoders follow the **BERT-style architecture** outlined in the project paper:

```
Input Modality → Pretrained Encoder → MLP Adapter → Semantic-Vector Space
```

**Key Components**:
1. **Pretrained Encoder**: Modality-specific model (BART, Whisper, ViT, etc.) that handles feature extraction
2. **MLP Adapter** (`utils.Adapter`): Small neural network that translates encoder outputs to the shared Semantic-Vector space

This design is **100x cheaper to train** than full fine-tuning because only the small MLP adapter is trained, not the large pretrained model.

## Current Encoders

### Text Encoder (`text_2_vec.py`)
- **Class**: `Text_to_Vec`
- **Base Model**: `facebook/bart-base`
- **Tokenizer**: `BertTokenizer`
- **Purpose**: Serves as the **ground truth** encoder for the semantic space
- **Architecture**: BART encoder → Adapter → Semantic Vector

### Audio Encoder (`wav_2_vec.py`)
- **Class**: `SEED` (or similar)
- **Base Model**: `openai/whisper-small`
- **Processor**: `WhisperProcessor`
- **Purpose**: Converts spoken language to semantic vectors
- **Architecture**: Whisper encoder → Adapter → Semantic Vector

### Image Encoder (`img_2_vec.py`)
- **Class**: `Image_to_Vec`
- **Base Model**: `nlpconnect/vit-gpt2-image-captioning`
- **Processor**: `ViTImageProcessor`
- **Purpose**: Converts images to semantic vectors
- **Architecture**: ViT encoder → Adapter → Semantic Vector

## How to Add a New Encoder

### Step 1: Create the Encoder Class

Create a new file `{modality}_2_vec.py` with the following structure:

```python
import torch
from transformers import <YourModel>, <YourProcessor>
from utils.Adapter import Adapter

BASE_MODEL = "huggingface/model-name"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class YourModality_to_Vec(torch.nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, token_length: int = 1024) -> None:
        super(YourModality_to_Vec, self).__init__()

        # 1. Initialize the processor/tokenizer
        self.processor = YourProcessor.from_pretrained(base_model, max_length=token_length)

        # 2. Create the MLP adapter
        # Adjust input_size/output_size based on your encoder's dimensions
        adapter = Adapter(
            prefix=f"{base_model}_enc",
            input_length=token_length,
            output_length=token_length,
            hidden_size=200,
            hidden_layers=2
        )

        # 3. Combine pretrained encoder with adapter
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            base_model,
            adapter
        )

    def forward(self, input_data, *, max_length: int = 1024, device: str = DEVICE):
        # 1. Process input to model format
        model_inputs = self.processor(
            input_data,
            return_tensors="pt",
            max_length=max_length
        ).to(device)

        # 2. Generate semantic vector
        output = self.model.generate(model_inputs)[0]

        return output
```

### Step 2: Configure the Adapter

The `Adapter` class parameters:
- **prefix**: Unique identifier for saving/loading weights
- **input_length**: Dimension of pretrained encoder output (default: 1024)
- **output_length**: Dimension of semantic vector space (default: 1024)
- **hidden_size**: Size of hidden layers in MLP (default: 200)
- **hidden_layers**: Number of hidden layers (default: 2)

### Step 3: Train the Encoder

Use the **Contrastive Learning** trainer in `src/Training/encoder_trainers.py`:

1. Prepare triplet data: `(query, positive, negative)`
   - Query: Your modality encoder output
   - Positive: Semantically similar sample (e.g., BART encoding of same text)
   - Negative: Semantically dissimilar sample

2. Use the `Contrast` trainer:
```python
from Training.encoder_trainers import Contrast

trainer = Contrast(
    model=your_encoder,
    # ... other Trainer arguments
)
trainer.train()
```

3. The trainer uses **triplet loss with cosine similarity**:
   - Loss = `max(cos_sim(query, positive) - cos_sim(query, negative) + 0.1, 0)`

4. Save adapter weights:
```python
your_encoder.adapter.save()  # Saves to AdapterWeights/{prefix}_weights.pth
```

### Step 4: Test Cross-Modal Alignment

Verify your encoder produces vectors similar to semantically-related content:
- Encode sample data from your modality
- Encode semantically-equivalent text using `text_2_vec.py`
- Measure cosine similarity (should be high for related content)

### Step 5: Add to Training Notebooks

Document your encoder training in:
- `src/encoder_training.ipynb` - Fine-tuning experiments
- `src/encoder_alignment.ipynb` - Alignment to semantic space

## Important Notes

1. **Ground Truth**: All encoders align to the semantic space defined by `facebook/bart-base` text encoder
2. **Backtranslation**: If you don't have direct (Modality → SemanticVector) pairs, use backtranslation:
   - Given: (Image, Text) and (Text, SemanticVector)
   - Synthesize: (Image, SemanticVector)
3. **Evaluation**: Test on unseen data with contrastive learning error and cross-modal similarity
4. **Device Support**: Always include CUDA/CPU detection for portability

## Archive

The `archive/` directory contains:
- **Old SEED implementations**: Previous encoder experiments
- **Legacy encoders**: `text/archive/` contains GPT.py, BART.py, novel.py
- **Experimental code**: VAE-based approaches and other historical implementations

**Do not modify archive code** unless explicitly working with legacy implementations.