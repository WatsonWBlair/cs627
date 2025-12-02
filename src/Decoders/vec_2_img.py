"""
EXPERIMENTAL: This decoder is not fully implemented.

Status: Under Development
Issues:
- Adapter integration with Stable Diffusion pipeline needs validation
- Training loop not tested
- Output format needs verification
- Text encoder replacement approach needs testing

DO NOT USE IN PRODUCTION.
"""

import torch
from diffusers import StableDiffusionPipeline
from torch import Generator
from PIL.Image import Image
from utils.Adapter import Adapter

BASE_MODEL: str = "CompVis/stable-diffusion-v1-4"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Vec_to_Image(torch.nn.Module):
    """
    EXPERIMENTAL: Image decoder using Adapter + Stable Diffusion.

    Architecture: Semantic Vector → Adapter → Stable Diffusion → Image

    Approach: Replaces the text encoder in Stable Diffusion with an adapter
    that accepts semantic vectors directly.

    Args:
        base_model: HuggingFace Stable Diffusion model (default: "CompVis/stable-diffusion-v1-4")
        output_dim: Semantic vector dimension (default: 1024)
    """

    def __init__(self, base_model: str = BASE_MODEL, output_dim: int = 1024) -> None:
        super(Vec_to_Image, self).__init__()

        # Adapter: translates FROM semantic space TO Stable Diffusion's text encoder space
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_dec",
            input_length=output_dim,  # Semantic vector size
            output_length=768,  # CLIP text encoder dimension
            hidden_size=200,
            hidden_layers=2
        )

        # Random number generator for sampling
        self.generator: Generator = Generator(device=DEVICE)

        # Load Stable Diffusion pipeline
        # Safety checker disabled for research use
        self.pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            safety_checker=None
        )

        # Replace text encoder with adapter
        # This allows direct semantic vector → image generation
        # TODO: Verify this approach produces semantically consistent images
        self.pipeline.text_encoder = self.adapter

        print(f"[Vec_to_Image] EXPERIMENTAL decoder initialized with {base_model}")
        print("[Vec_to_Image] WARNING: Text encoder replaced with adapter - needs validation")

    def forward(self, semantic_vector: torch.Tensor, *, device: str = DEVICE) -> Image:
        """
        Generate image from semantic vector.

        Args:
            semantic_vector: Tensor of shape (batch_size, output_dim)
            device: Device to run on

        Returns:
            Generated PIL Image

        Note: Current implementation is experimental.
        TODO: Validate that semantic vectors produce meaningful images
        """
        # Ensure input is on correct device
        semantic_vector = semantic_vector.to(device)
        self.pipeline = self.pipeline.to(device)
        self.generator = Generator(device=device)

        # Generate image directly from semantic vector
        # The adapter replaces the text encoder, so semantic_vector goes directly to diffusion
        output = self.pipeline(
            prompt_embeds=semantic_vector,  # Use semantic vector as prompt embeddings
            generator=self.generator,
            num_inference_steps=50
        ).images[0]

        return output
