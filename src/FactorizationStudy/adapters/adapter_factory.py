"""
Adapter Factory for creating adapters with custom configurations.

This module provides factory methods to create Adapter instances
with configurable hidden sizes and layer counts for the factorization study.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils.Adapter import Adapter
from src.FactorizationStudy.config.study_config import AdapterConfig


class ConfigurableAdapter(nn.Module):
    """
    Extended Adapter with additional configuration options for factorization study.

    Adds:
    - Configurable initialization
    - Weight transfer from other adapters
    - Dimension tracking for study metrics
    """

    def __init__(
        self,
        prefix: str,
        input_length: int,
        output_length: int,
        config: AdapterConfig,
        weights_dir: str = "OptimalWeights",
    ) -> None:
        """
        Initialize a configurable adapter.

        Args:
            prefix: Unique identifier for saving/loading weights
            input_length: Input dimension (from pretrained model)
            output_length: Output dimension (semantic space)
            config: AdapterConfig with hidden_size and hidden_layers
            weights_dir: Directory to save/load weights
        """
        super().__init__()
        self.prefix = prefix
        self.input_length = input_length
        self.output_length = output_length
        self.config = config
        self.weights_dir = weights_dir
        self.weights_path = Path(f"{weights_dir}/{prefix}_weights.pth")

        # Build MLP layers
        layers: list[nn.Module] = [
            nn.Linear(input_length, config.hidden_size),
            nn.ReLU()
        ]

        for _ in range(config.hidden_layers):
            layers.extend([
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU()
            ])

        layers.append(nn.Linear(config.hidden_size, output_length))

        self.model = nn.Sequential(*layers)

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter."""
        return self.model(input_vectors)

    def load(self, device: Optional[str] = None) -> bool:
        """
        Load adapter weights from disk.

        Args:
            device: Device to load weights to

        Returns:
            True if weights loaded successfully, False otherwise
        """
        if not self.weights_path.exists():
            return False

        weights = torch.load(self.weights_path, map_location=device or 'cpu')
        self.model.load_state_dict(weights)
        return True

    def save(self) -> None:
        """Save adapter weights to disk."""
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)

    def initialize_random(self, seed: Optional[int] = None) -> None:
        """Reinitialize weights randomly."""
        if seed is not None:
            torch.manual_seed(seed)

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def transfer_from(self, source_adapter: "ConfigurableAdapter") -> bool:
        """
        Transfer weights from another adapter.

        Only transfers if architectures are compatible (same hidden_size and hidden_layers).

        Args:
            source_adapter: Adapter to transfer weights from

        Returns:
            True if transfer successful, False if architectures incompatible
        """
        if (self.config.hidden_size != source_adapter.config.hidden_size or
            self.config.hidden_layers != source_adapter.config.hidden_layers):
            return False

        # Check input/output dimensions match
        if (self.input_length != source_adapter.input_length or
            self.output_length != source_adapter.output_length):
            return False

        # Transfer weights
        self.model.load_state_dict(source_adapter.model.state_dict())
        return True

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_info(self) -> Dict[str, Any]:
        """Get adapter information for logging."""
        return {
            "prefix": self.prefix,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "hidden_size": self.config.hidden_size,
            "hidden_layers": self.config.hidden_layers,
            "parameters": self.count_parameters(),
        }


class AdapterFactory:
    """
    Factory for creating adapters with various configurations.
    """

    # Known pretrained model hidden dimensions
    MODEL_HIDDEN_DIMS = {
        "facebook/bart-base": 768,
        "openai/whisper-small": 768,
        "nlpconnect/vit-gpt2-image-captioning": 768,
        "microsoft/wavlm-base": 768,
    }

    @classmethod
    def create_adapter(
        cls,
        prefix: str,
        input_length: int,
        output_length: int,
        config: AdapterConfig,
        weights_dir: str = "OptimalWeights",
    ) -> ConfigurableAdapter:
        """
        Create a configurable adapter with specified parameters.

        Args:
            prefix: Unique identifier
            input_length: Input dimension
            output_length: Output dimension (semantic space)
            config: Adapter configuration
            weights_dir: Directory for weights

        Returns:
            ConfigurableAdapter instance
        """
        return ConfigurableAdapter(
            prefix=prefix,
            input_length=input_length,
            output_length=output_length,
            config=config,
            weights_dir=weights_dir,
        )

    @classmethod
    def create_text_adapter(
        cls,
        semantic_dim: int,
        config: AdapterConfig,
        weights_dir: str = "OptimalWeights",
        prefix_suffix: str = "",
    ) -> ConfigurableAdapter:
        """Create an adapter for text encoding (BART-based)."""
        prefix = f"facebook_bart-base_text_enc{prefix_suffix}"
        return cls.create_adapter(
            prefix=prefix,
            input_length=cls.MODEL_HIDDEN_DIMS["facebook/bart-base"],
            output_length=semantic_dim,
            config=config,
            weights_dir=weights_dir,
        )

    @classmethod
    def create_audio_adapter(
        cls,
        semantic_dim: int,
        config: AdapterConfig,
        weights_dir: str = "OptimalWeights",
        prefix_suffix: str = "",
    ) -> ConfigurableAdapter:
        """Create an adapter for audio encoding (Whisper-based)."""
        prefix = f"openai_whisper-small_audio_enc{prefix_suffix}"
        return cls.create_adapter(
            prefix=prefix,
            input_length=cls.MODEL_HIDDEN_DIMS["openai/whisper-small"],
            output_length=semantic_dim,
            config=config,
            weights_dir=weights_dir,
        )

    @classmethod
    def create_image_adapter(
        cls,
        semantic_dim: int,
        config: AdapterConfig,
        weights_dir: str = "OptimalWeights",
        prefix_suffix: str = "",
    ) -> ConfigurableAdapter:
        """Create an adapter for image encoding (ViT-based)."""
        prefix = f"nlpconnect_vit-gpt2-image-captioning_image_enc{prefix_suffix}"
        return cls.create_adapter(
            prefix=prefix,
            input_length=cls.MODEL_HIDDEN_DIMS["nlpconnect/vit-gpt2-image-captioning"],
            output_length=semantic_dim,
            config=config,
            weights_dir=weights_dir,
        )

    @classmethod
    def get_model_hidden_dim(cls, model_name: str) -> int:
        """Get the hidden dimension for a known pretrained model."""
        return cls.MODEL_HIDDEN_DIMS.get(model_name, 768)
