"""
Decoder Boilerplate Template

This is a template for creating new decoders. To use:
1. Copy this file to vec_2_{modality}.py
2. Replace all {Modality} and {MODALITY} placeholders
3. Update BASE_MODEL to your HuggingFace generative model
4. Update processor/model classes from transformers
5. Customize forward() method for your modality
6. Implement proper generation logic

IMPORTANT: You MUST have a trained encoder for this modality before creating a decoder!
The encoder is required for backtranslation training.

Example:
    cp src/Decoders/decoder_boilerplate.py src/Decoders/vec_2_video.py
    # Then edit vec_2_video.py to implement video generation
"""

import torch
from transformers import AutoModel, AutoProcessor  # TODO: Replace with specific classes
from typing import Optional
from utils.Adapter import Adapter

# TODO: Update these constants
BASE_MODEL: str = "huggingface/generative-model-name"  # e.g., "facebook/bart-base"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Vec_to_{Modality}(torch.nn.Module):
    """
    {Modality} decoder using Adapter + {ModelName}.

    Architecture: Semantic Vector → Adapter → {ModelName} Decoder → {Modality} Output

    Prerequisites:
        - Trained {Modality}_to_Vec encoder must exist
        - Paired training data for backtranslation

    Args:
        base_model: Pretrained generative model identifier (default: BASE_MODEL)
        output_dim: Semantic vector dimension (default: 1024)

    Example:
        >>> decoder = Vec_to_{Modality}()
        >>> semantic_vec = torch.randn(1, 1024)
        >>> output = decoder(semantic_vec)
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        output_dim: int = 1024
    ) -> None:
        super(Vec_to_{Modality}, self).__init__()

        # TODO: Replace AutoProcessor with specific processor/tokenizer class
        # Examples: BartTokenizer, AutoProcessor, ViTImageProcessor
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(base_model)

        # Adapter: translates FROM semantic space TO decoder input space
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_{modality}_dec",
            input_length=output_dim,  # Always 1024 from semantic space
            output_length=output_dim,  # TODO: Adjust if decoder needs different dim
            hidden_size=200,
            hidden_layers=2
        )

        # TODO: Replace AutoModel with specific generative model class
        # Examples: BartForConditionalGeneration, StableDiffusionPipeline, etc.
        self.decoder: AutoModel = AutoModel.from_pretrained(base_model)

        print(f"[Vec_to_{Modality}] Decoder initialized with {base_model}")

    def forward(
        self,
        semantic_vector: torch.Tensor,
        *,
        device: str = DEVICE,
        **generation_kwargs  # Additional args for model.generate()
    ) -> ...:  # TODO: Specify return type (str, torch.Tensor, PIL.Image, etc.)
        """
        Generate {modality} output from semantic vector.

        Args:
            semantic_vector: Tensor of shape (batch_size, output_dim)
            device: Device to run on
            **generation_kwargs: Additional arguments passed to model.generate()
                Common options: max_length, num_beams, temperature, top_k, top_p

        Returns:
            Generated {modality} output
            # TODO: Specify exact return type and format

        Example:
            >>> decoder = Vec_to_{Modality}()
            >>> vec = torch.randn(1, 1024)
            >>> output = decoder(vec, max_length=50)
        """
        # Ensure input is on correct device
        semantic_vector = semantic_vector.to(device)
        self.adapter = self.adapter.to(device)
        self.decoder = self.decoder.to(device)

        # Project semantic vector to decoder input space
        decoder_input: torch.Tensor = self.adapter(semantic_vector)

        # TODO: Generate output using decoder
        # Different approaches depending on model type:

        # APPROACH 1: Text generation (BART, GPT, T5)
        # output_ids = self.decoder.generate(
        #     inputs_embeds=decoder_input.unsqueeze(1),
        #     max_length=generation_kwargs.get('max_length', 50),
        #     num_beams=generation_kwargs.get('num_beams', 5),
        #     early_stopping=True
        # )
        # output = self.processor.batch_decode(output_ids, skip_special_tokens=True)

        # APPROACH 2: Image generation (Stable Diffusion)
        # from diffusers import StableDiffusionPipeline
        # self.decoder is a pipeline
        # output = self.decoder(
        #     prompt_embeds=decoder_input,
        #     num_inference_steps=50
        # ).images[0]

        # APPROACH 3: Audio generation (TTS)
        # from transformers import pipeline
        # Need intermediate text step or direct embedding approach
        # output = self.decoder(decoder_input)

        # TODO: Implement generation logic for your modality
        raise NotImplementedError(
            "Generation logic not yet implemented. "
            "See comments above for common approaches."
        )

        # TODO: Post-process output if needed
        # - Text: Decode token IDs to strings
        # - Image: Denormalize, convert to PIL
        # - Audio: Convert to waveform, resample

        # TODO: Handle batch size 1 vs batch
        # if isinstance(output, list) and len(output) == 1:
        #     return output[0]
        # return output


# TODO: Add usage example
if __name__ == "__main__":
    print(f"Testing Vec_to_{Modality} decoder...")

    # Check if corresponding encoder exists
    try:
        from Encoders.{modality}_2_vec import {Modality}_to_Vec
        print(f"✓ Corresponding encoder found: {Modality}_to_Vec")
    except ImportError:
        print(f"✗ WARNING: No encoder found at Encoders/{modality}_2_vec.py")
        print(f"  A decoder requires a trained encoder for backtranslation!")

    # Initialize decoder
    decoder = Vec_to_{Modality}()

    # TODO: Test with random semantic vector
    sample_vec = torch.randn(1, 1024)

    try:
        output = decoder(sample_vec)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {sample_vec.shape}")
        print(f"  Output type: {type(output)}")
        # TODO: Add output shape/format validation
    except NotImplementedError:
        print(f"⚠ Forward pass not implemented yet (expected)")
        print(f"  Implement generation logic in forward() method")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

    # Test adapter save/load
    try:
        decoder.adapter.save()
        print(f"✓ Adapter weights saved")
        decoder.adapter.load()
        print(f"✓ Adapter weights loaded")
    except Exception as e:
        print(f"✗ Error saving/loading weights: {e}")

    print(f"\nVec_to_{Modality} decoder test complete!")
    print(f"\nNext steps:")
    print(f"1. Implement generation logic in forward()")
    print(f"2. Test with real semantic vectors from encoder")
    print(f"3. Implement backtranslation training loop")
    print(f"4. Evaluate reconstruction quality")
