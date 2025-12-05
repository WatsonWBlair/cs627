"""
Image_to_Vec: Image Encoder using ViT + MLP Adapter

Architecture: PIL Image -> ViT Encoder -> Mean Pooling -> Adapter -> Semantic Vector (1024)

Status: PRODUCTION
- Uses nlpconnect/vit-gpt2-image-captioning ViT encoder
- Adapter projects from ViT hidden dim (768) to semantic space (1024)
- Compatible with Vec_to_Image decoder for round-trip encoding

Based on: nlpconnect/vit-gpt2-image-captioning

Usage:
    from src.Encoders import Image_to_Vec
    from PIL import Image

    encoder = Image_to_Vec()
    image = Image.open("photo.jpg")
    semantic_vec = encoder(image)  # Returns (1, 1024) tensor
"""

import torch
from transformers import ViTImageProcessor, VisionEncoderDecoderModel
from typing import Union
from PIL.Image import Image
from src.utils.Adapter import Adapter

# Model identifiers
BASE_MODEL: str = "nlpconnect/vit-gpt2-image-captioning"
VIT_BASE_HIDDEN_DIM: int = 768  # ViT-base encoder output dimension

# Device configuration
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Image_to_Vec(torch.nn.Module):
    """
    Image encoder using ViT + MLP Adapter.

    Architecture:
        PIL Image -> ViT Encoder -> Mean Pooling (768) -> Adapter -> Semantic Vector (1024)

    The ViT encoder extracts visual features as a sequence of patch embeddings.
    Mean pooling aggregates these into a single 768-dimensional vector.
    The adapter then projects to the shared 1024-dimensional semantic space.

    Args:
        base_model: HuggingFace ViT model identifier (default: "nlpconnect/vit-gpt2-image-captioning")
        output_dim: Semantic vector space dimension (default: 1024)
        freeze_encoder: Whether to freeze ViT encoder weights (default: True)
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        output_dim: int = 1024,
        freeze_encoder: bool = True
    ) -> None:
        super(Image_to_Vec, self).__init__()

        # Store configuration
        self.base_model_name = base_model

        # Load image processor
        print(f"[Image_to_Vec] Loading ViT from {base_model}...")
        self.processor: ViTImageProcessor = ViTImageProcessor.from_pretrained(base_model)

        # Load vision encoder-decoder model
        self.model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(base_model)

        # Freeze pretrained encoder to prevent catastrophic forgetting
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print(f"[Image_to_Vec] ViT encoder frozen (only adapter trainable)")

        # Create adapter: ViT hidden dim (768) → Semantic vector space (1024)
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_image_enc",
            input_length=VIT_BASE_HIDDEN_DIM,
            output_length=output_dim,
            hidden_size=200,
            hidden_layers=2
        )

        # Track trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Image_to_Vec] Encoder initialized with {base_model}")
        print(f"[Image_to_Vec] Adapter: {VIT_BASE_HIDDEN_DIM} -> {output_dim}")
        print(f"[Image_to_Vec] Adapter has {trainable:,} trainable parameters")

        # Try to load trained adapter weights
        try:
            self.adapter.load(device=DEVICE)
            print(f"[Image_to_Vec] [OK] Loaded adapter weights: {self.adapter.weights_path}")
        except FileNotFoundError:
            print(f"[Image_to_Vec] [WARNING] No trained adapter weights found!")
            print(f"[Image_to_Vec] [WARNING] Expected: {self.adapter.weights_path}")
            print(f"[Image_to_Vec] [WARNING] Encoder will NOT work until trained.")
            print(f"[Image_to_Vec] [WARNING] Train using: python src/Training/train_encoders.py")

    def forward(
        self,
        input_img: Union[Image, list],
        *,
        device: str = DEVICE
    ) -> torch.Tensor:
        """
        Forward pass: image -> ViT encoder -> mean pooling -> adapter -> semantic vector

        Args:
            input_img: PIL Image or list of PIL Images
            device: Device to run on

        Returns:
            Semantic vector representation (batch_size, 1024)
        """
        # Move model components to device
        self.model = self.model.to(device)
        self.adapter = self.adapter.to(device)

        # Process image input to pixel values
        pixel_values = self.processor(input_img, return_tensors="pt").pixel_values.to(device)

        # Extract continuous embeddings from vision encoder (not discrete tokens)
        with torch.set_grad_enabled(
            self.model.encoder.training and
            any(p.requires_grad for p in self.model.encoder.parameters())
        ):
            encoder_outputs = self.model.encoder(pixel_values)

        # Get last hidden state and apply mean pooling
        hidden_states = encoder_outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_dim)
        pooled_output = hidden_states.mean(dim=1)  # Shape: (batch, hidden_dim)

        # Convert to semantic vector via adapter
        semantic_vector = self.adapter(pooled_output)

        return semantic_vector