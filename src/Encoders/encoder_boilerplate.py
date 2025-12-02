"""
Encoder Boilerplate Template

This is a template for creating new encoders. To use:
1. Copy this file to {modality}_2_vec.py
2. Replace all {Modality} and {MODALITY} placeholders
3. Update BASE_MODEL to your HuggingFace model
4. Adjust ENCODER_DIM based on your model's output
5. Update processor/model classes from transformers
6. Customize forward() method for your modality

Example:
    cp src/Encoders/encoder_boilerplate.py src/Encoders/video_2_vec.py
    # Then edit video_2_vec.py to implement video encoding
"""

import torch
from transformers import AutoModel, AutoProcessor  # TODO: Replace with specific classes
from typing import Union, Optional, List
from utils.Adapter import Adapter

# TODO: Update these constants
BASE_MODEL: str = "huggingface/model-name"  # e.g., "facebook/bart-base"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: Determine your model's encoder output dimension
# Common values: BART=768, Whisper-small=768, BERT-base=768, ViT varies
{MODALITY}_ENCODER_DIM = 768  # CHANGEME: Check model.config.hidden_size


class {Modality}_to_Vec(torch.nn.Module):
    """
    {Modality} encoder using {ModelName} + MLP Adapter.

    Architecture: {Modality} Input → {ModelName} Encoder → Mean Pool → Adapter → Semantic Vector

    Args:
        base_model: Pretrained model identifier (default: BASE_MODEL)
        output_dim: Semantic vector dimension (default: 1024)
        freeze_encoder: Whether to freeze pretrained encoder (default: True)

    Example:
        >>> encoder = {Modality}_to_Vec()
        >>> semantic_vec = encoder(input_data)
        >>> print(semantic_vec.shape)  # (batch_size, 1024)
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        output_dim: int = 1024,
        freeze_encoder: bool = True
    ) -> None:
        super({Modality}_to_Vec, self).__init__()

        # TODO: Replace AutoProcessor with specific processor class
        # Examples: BartTokenizer, WhisperProcessor, ViTImageProcessor, etc.
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(base_model)

        # TODO: Replace AutoModel with specific model class
        # Examples: BartModel, WhisperForConditionalGeneration, VisionEncoderDecoderModel
        self.model = AutoModel.from_pretrained(base_model)

        # TODO: Extract encoder component (if model has separate encoder)
        # Some models: self.encoder = self.model.get_encoder()
        # Others: self.encoder = self.model (if model is already encoder-only)
        self.encoder = self.model.get_encoder()  # CHANGEME based on your model

        # Freeze pretrained encoder to prevent catastrophic forgetting
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            print(f"[{Modality}_to_Vec] Encoder frozen (0 trainable params in encoder)")

        # Create adapter: maps encoder output → semantic space
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_{modality}_enc",  # Unique identifier
            input_length={MODALITY}_ENCODER_DIM,  # TODO: Verify this matches your model
            output_length=output_dim,  # Always 1024 for semantic space
            hidden_size=200,
            hidden_layers=2
        )

        # Print trainable parameter count
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{Modality}_to_Vec] Adapter has {trainable_params:,} trainable parameters")

    def forward(
        self,
        input_data: Union[...],  # TODO: Specify input type (str, torch.Tensor, PIL.Image, etc.)
        *,
        device: str = DEVICE
    ) -> torch.Tensor:
        """
        Encode {modality} input to semantic vector.

        Args:
            input_data: {Description of input format}
                - Single: {format description}
                - Batch: {format description}
            device: Device to run on

        Returns:
            Semantic vectors of shape (batch_size, output_dim)

        Example:
            >>> encoder = {Modality}_to_Vec()
            >>> vec = encoder(sample_input)
            >>> print(vec.shape)  # torch.Size([1, 1024])
        """
        # Move encoder to device
        self.encoder = self.encoder.to(device)

        # TODO: Process input to model format
        # Adjust preprocessing based on your modality:
        # - Text: tokenization with max_length, truncation, padding
        # - Audio: resampling, spectrogram conversion
        # - Image: resize, normalization
        # - Video: frame sampling, normalization
        processed_input = self.processor(
            input_data,
            return_tensors="pt",
            # TODO: Add modality-specific preprocessing args
            # Examples: max_length=1024, sampling_rate=16000, size=(224, 224)
        ).to(device)

        # Pass through encoder (gradient control for frozen encoder)
        with torch.set_grad_enabled(
            self.encoder.training and
            any(p.requires_grad for p in self.encoder.parameters())
        ):
            encoder_outputs = self.encoder(**processed_input)

            # TODO: Extract hidden states from encoder output
            # Common patterns:
            # - encoder_outputs.last_hidden_state  # Most transformers
            # - encoder_outputs[0]  # Some older models
            # - encoder_outputs.hidden_states[-1]  # With output_hidden_states=True
            hidden_states = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # TODO: Pool to fixed-size representation
        # Pooling strategies:
        # - Mean pooling: hidden_states.mean(dim=1)  # Average over sequence
        # - Max pooling: hidden_states.max(dim=1)[0]  # Max over sequence
        # - CLS token: hidden_states[:, 0, :]  # First token (BERT-style)
        # - Last token: hidden_states[:, -1, :]  # Final token
        pooled_output: torch.Tensor = hidden_states.mean(dim=1)  # (batch, hidden_dim)

        # Project to semantic space via adapter
        semantic_vector: torch.Tensor = self.adapter(pooled_output)

        return semantic_vector


# TODO: Add usage example
if __name__ == "__main__":
    print(f"Testing {Modality}_to_Vec encoder...")

    # Initialize encoder
    encoder = {Modality}_to_Vec()

    # TODO: Create sample input for your modality
    # Examples:
    # - Text: sample_input = "This is a test sentence."
    # - Audio: sample_input = torch.randn(1, 16000)  # 1 second at 16kHz
    # - Image: sample_input = PIL.Image.new('RGB', (224, 224))
    sample_input = ...  # CHANGEME: Provide realistic sample input

    # Test forward pass
    try:
        semantic_vec = encoder(sample_input)
        print(f"✓ Forward pass successful")
        print(f"  Input: {type(sample_input)}")
        print(f"  Output shape: {semantic_vec.shape}")
        print(f"  Expected: (batch_size, 1024)")
        assert semantic_vec.shape[-1] == 1024, "Output dimension should be 1024"
        print(f"✓ Output dimension correct")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

    # Test adapter save/load
    try:
        encoder.adapter.save()
        print(f"✓ Adapter weights saved")
        encoder.adapter.load()
        print(f"✓ Adapter weights loaded")
    except Exception as e:
        print(f"✗ Error saving/loading weights: {e}")

    print(f"\n{Modality}_to_Vec encoder test complete!")
