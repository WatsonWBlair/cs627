"""
Vec_to_Image: Semantic Vector to Image Decoder using Stable Diffusion

Architecture: Semantic Vector (1024) -> Adapter (77×768) -> Stable Diffusion -> Image

Status: EXPERIMENTAL
- Learned sequence adapter projects semantic vector to full CLIP embedding sequence
- Stable Diffusion v1.4 generates images from prompt embeddings
- Training pipeline: python src/Training/train_adapters.py --mode decoder

Based on: CompVis/stable-diffusion-v1-4

Usage:
    from src.Decoders import Vec_to_Image
    from src.Encoders import Text_to_Vec

    # Encode text to semantic vector
    encoder = Text_to_Vec()
    semantic_vec = encoder("A beautiful sunset over the ocean")

    # Decode to image
    decoder = Vec_to_Image()
    image = decoder(semantic_vec)  # Returns PIL Image
    image.save("output.png")
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from torch import Generator
from PIL.Image import Image
from typing import Optional, Union, List
import numpy as np

from src.utils.Adapter import Adapter

# Model identifiers
BASE_MODEL: str = "CompVis/stable-diffusion-v1-4"

# CLIP text encoder specifications (for Stable Diffusion v1.x)
CLIP_SEQ_LENGTH: int = 77  # Maximum token sequence length
CLIP_HIDDEN_DIM: int = 768  # CLIP ViT-L/14 hidden dimension
FULL_EMBED_DIM: int = CLIP_SEQ_LENGTH * CLIP_HIDDEN_DIM  # 59,136

# Semantic vector dimension
SEMANTIC_DIM: int = 1024

# Device configuration
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Vec_to_Image(nn.Module):
    """
    Image decoder using Adapter + Stable Diffusion.

    Architecture:
        Semantic Vector (1024) -> Adapter MLP -> (77×768) -> Reshape -> Stable Diffusion -> PIL Image

    The adapter learns to project a 1024-dimensional semantic vector to a full
    77-token CLIP embedding sequence (77 × 768 = 59,136 dimensions), which is
    then reshaped and passed directly to Stable Diffusion as prompt_embeds.

    Args:
        base_model: HuggingFace Stable Diffusion model identifier
        input_dim: Semantic vector dimension (default: 1024)
        freeze_pipeline: Whether to freeze SD pipeline weights (default: True)
        guidance_scale: Default classifier-free guidance scale (default: 7.5)
        num_inference_steps: Default denoising steps (default: 50)
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        input_dim: int = SEMANTIC_DIM,
        freeze_pipeline: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> None:
        super(Vec_to_Image, self).__init__()

        # Store configuration
        self.base_model_name = base_model
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        # Adapter: Semantic space (1024) -> Full CLIP sequence (77 × 768 = 59,136)
        # Larger hidden size and more layers for complex mapping
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_image_dec",
            input_length=input_dim,
            output_length=FULL_EMBED_DIM,  # 77 × 768 = 59,136
            hidden_size=512,  # Larger for 1024 -> 59136 projection
            hidden_layers=3   # More depth for sequence learning
        )

        # Random number generator for reproducibility
        self.generator: Generator = Generator(device="cpu")

        # Load Stable Diffusion pipeline
        print(f"[Vec_to_Image] Loading Stable Diffusion from {base_model}...")
        self.pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            safety_checker=None,  # Disabled for research use
            torch_dtype=torch.float32
        )

        # Freeze pipeline components (only adapter is trainable)
        if freeze_pipeline:
            # Freeze UNet
            for param in self.pipeline.unet.parameters():
                param.requires_grad = False

            # Freeze VAE
            for param in self.pipeline.vae.parameters():
                param.requires_grad = False

            # Freeze text encoder (we don't use it but keep it frozen)
            if self.pipeline.text_encoder is not None:
                for param in self.pipeline.text_encoder.parameters():
                    param.requires_grad = False

            print(f"[Vec_to_Image] Pipeline frozen (only adapter trainable)")

        # Track trainable parameters
        trainable = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"[Vec_to_Image] Decoder initialized with {base_model}")
        print(f"[Vec_to_Image] Adapter: {input_dim} -> {FULL_EMBED_DIM} ({CLIP_SEQ_LENGTH}×{CLIP_HIDDEN_DIM})")
        print(f"[Vec_to_Image] Adapter has {trainable:,} trainable parameters")

        # Try to load trained adapter weights
        try:
            self.adapter.load(device=DEVICE)
            print(f"[Vec_to_Image] [OK] Loaded adapter weights: {self.adapter.weights_path}")
        except FileNotFoundError:
            print(f"[Vec_to_Image] [WARNING] No trained adapter weights found!")
            print(f"[Vec_to_Image] [WARNING] Expected: {self.adapter.weights_path}")
            print(f"[Vec_to_Image] [WARNING] Decoder will NOT produce meaningful images until trained.")
            print(f"[Vec_to_Image] [WARNING] Train using: python src/Training/train_adapters.py --mode decoder")

    def forward(
        self,
        semantic_vector: torch.Tensor,
        *,
        device: str = DEVICE,
        height: int = 512,
        width: int = 512,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Union[Image, List[Image]]:
        """
        Generate image(s) from semantic vector(s).

        Args:
            semantic_vector: Tensor of shape (batch_size, 1024) or (1024,)
            device: Device to run generation on
            height: Output image height (default: 512)
            width: Output image width (default: 512)
            guidance_scale: Classifier-free guidance scale (default: self.guidance_scale)
            num_inference_steps: Number of denoising steps (default: self.num_inference_steps)
            seed: Random seed for reproducibility (default: None)

        Returns:
            PIL Image for batch_size=1, or list of PIL Images for batch_size>1
        """
        # Handle 1D input
        if semantic_vector.dim() == 1:
            semantic_vector = semantic_vector.unsqueeze(0)

        batch_size = semantic_vector.shape[0]

        # Move components to device
        semantic_vector = semantic_vector.to(device)
        self.adapter = self.adapter.to(device)
        self.pipeline = self.pipeline.to(device)

        # Set up generator for reproducibility
        if seed is not None:
            self.generator = Generator(device="cpu").manual_seed(seed)

        # Use default parameters if not specified
        guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps

        # 1. Adapter: semantic vector -> flat CLIP embedding sequence
        with torch.no_grad():
            flat_embeds = self.adapter(semantic_vector)  # (batch, 59136)

        # 2. Reshape to prompt embeddings (batch, 77, 768)
        prompt_embeds = flat_embeds.reshape(batch_size, CLIP_SEQ_LENGTH, CLIP_HIDDEN_DIM)

        # 3. Create negative prompt embeddings (zeros for neutral baseline)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        # 4. Generate images using Stable Diffusion
        with torch.no_grad():
            output = self.pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                height=height,
                width=width,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=self.generator
            )

        # 5. Return image(s)
        images = output.images

        if batch_size == 1:
            return images[0]
        return images

    def generate_with_latents(
        self,
        semantic_vector: torch.Tensor,
        *,
        device: str = DEVICE,
        height: int = 512,
        width: int = 512,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None
    ) -> tuple:
        """
        Generate image with intermediate latents (for training/analysis).

        Args:
            semantic_vector: Tensor of shape (batch_size, 1024) or (1024,)
            device: Device to run generation on
            height: Output image height
            width: Output image width
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility

        Returns:
            Tuple of (PIL Image(s), prompt_embeds tensor)
        """
        # Handle 1D input
        if semantic_vector.dim() == 1:
            semantic_vector = semantic_vector.unsqueeze(0)

        batch_size = semantic_vector.shape[0]

        # Move components to device
        semantic_vector = semantic_vector.to(device)
        self.adapter = self.adapter.to(device)
        self.pipeline = self.pipeline.to(device)

        # Set up generator
        if seed is not None:
            self.generator = Generator(device="cpu").manual_seed(seed)

        # Use default parameters
        guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps

        # 1. Adapter forward (keep for return)
        flat_embeds = self.adapter(semantic_vector)  # (batch, 59136)
        prompt_embeds = flat_embeds.reshape(batch_size, CLIP_SEQ_LENGTH, CLIP_HIDDEN_DIM)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        # 2. Generate
        with torch.no_grad():
            output = self.pipeline(
                prompt_embeds=prompt_embeds.detach(),
                negative_prompt_embeds=negative_prompt_embeds.detach(),
                height=height,
                width=width,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=self.generator
            )

        images = output.images
        if batch_size == 1:
            return images[0], prompt_embeds

        return images, prompt_embeds

    def get_adapter_output(
        self,
        semantic_vector: torch.Tensor,
        *,
        device: str = DEVICE
    ) -> torch.Tensor:
        """
        Get adapter output for training (differentiable path).

        This method is used during training to get the adapter output
        before the non-differentiable image generation step.

        Args:
            semantic_vector: Tensor of shape (batch_size, 1024)
            device: Device to run on

        Returns:
            Flat embedding tensor of shape (batch_size, 59136)
        """
        semantic_vector = semantic_vector.to(device)
        self.adapter = self.adapter.to(device)

        # Adapter forward pass (differentiable)
        flat_embeds = self.adapter(semantic_vector)

        return flat_embeds

    def get_prompt_embeds(
        self,
        semantic_vector: torch.Tensor,
        *,
        device: str = DEVICE
    ) -> torch.Tensor:
        """
        Get reshaped prompt embeddings (differentiable).

        Args:
            semantic_vector: Tensor of shape (batch_size, 1024)
            device: Device to run on

        Returns:
            Prompt embeddings of shape (batch_size, 77, 768)
        """
        flat_embeds = self.get_adapter_output(semantic_vector, device=device)
        batch_size = semantic_vector.shape[0] if semantic_vector.dim() > 1 else 1
        prompt_embeds = flat_embeds.reshape(batch_size, CLIP_SEQ_LENGTH, CLIP_HIDDEN_DIM)

        return prompt_embeds

    def encode_image_to_latent(
        self,
        image: Image,
        *,
        device: str = DEVICE
    ) -> torch.Tensor:
        """
        Encode a PIL image to VAE latent space (for training loss).

        Args:
            image: PIL Image to encode
            device: Device to run on

        Returns:
            Latent tensor from VAE encoder
        """
        from torchvision import transforms

        # Preprocess image
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
        ])

        # Convert to tensor
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # Encode with VAE
        self.pipeline = self.pipeline.to(device)
        with torch.no_grad():
            latent_dist = self.pipeline.vae.encode(image_tensor)
            latents = latent_dist.latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor

        return latents
