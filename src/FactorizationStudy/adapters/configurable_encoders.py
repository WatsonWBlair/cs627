"""
Configurable Encoders for Factorization Study.

These wrappers around the production encoders allow runtime configuration of:
- Output dimension (semantic_dim)
- Adapter hidden size
- Adapter hidden layers

They reuse the pretrained backbones but create new adapters with custom configurations.
"""

import torch
import torch.nn as nn
from typing import Union, Optional
from pathlib import Path

# Transformers imports
from transformers import (
    BartModel, BartTokenizer,
    WhisperProcessor, WhisperForConditionalGeneration,
    ViTImageProcessor, VisionEncoderDecoderModel,
)

try:
    from PIL.Image import Image
except ImportError:
    Image = type(None)

from .adapter_factory import ConfigurableAdapter, AdapterFactory
from ..config.study_config import AdapterConfig, InitStrategy

# Device configuration
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Model hidden dimensions
BART_BASE_HIDDEN_DIM = 768
WHISPER_SMALL_ENCODER_DIM = 768
VIT_BASE_HIDDEN_DIM = 768


class ConfigurableTextEncoder(nn.Module):
    """
    Configurable text encoder using BART + custom MLP Adapter.

    Allows runtime configuration of semantic dimension and adapter architecture.
    """

    def __init__(
        self,
        semantic_dim: int = 1024,
        adapter_config: Optional[AdapterConfig] = None,
        base_model: str = "facebook/bart-base",
        max_length: int = 1024,
        weights_dir: str = "OptimalWeights",
        experiment_id: str = "",
        init_strategy: InitStrategy = InitStrategy.PRETRAINED,
        load_weights: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Initialize configurable text encoder.

        Args:
            semantic_dim: Output semantic vector dimension
            adapter_config: Adapter configuration (hidden_size, hidden_layers)
            base_model: HuggingFace BART model identifier
            max_length: Maximum sequence length
            weights_dir: Directory for adapter weights
            experiment_id: Unique identifier for this experiment's weights
            init_strategy: How to initialize adapter weights
            load_weights: Whether to load pretrained weights if available
            verbose: Whether to print status messages
        """
        super().__init__()

        self.semantic_dim = semantic_dim
        self.adapter_config = adapter_config or AdapterConfig()
        self.max_length = max_length
        self.base_model_name = base_model
        self.verbose = verbose

        # Load tokenizer and encoder
        if verbose:
            print(f"[ConfigurableTextEncoder] Loading BART from {base_model}...")

        self.tokenizer = BartTokenizer.from_pretrained(base_model)
        self.encoder = BartModel.from_pretrained(base_model)

        # Freeze pretrained encoder
        self.encoder.requires_grad_(False)

        # Create configurable adapter
        prefix = f"{base_model.replace('/', '_')}_text_enc"
        if experiment_id:
            prefix = f"{prefix}_{experiment_id}"

        self.adapter = ConfigurableAdapter(
            prefix=prefix,
            input_length=BART_BASE_HIDDEN_DIM,
            output_length=semantic_dim,
            config=self.adapter_config,
            weights_dir=weights_dir,
        )

        # Initialize weights based on strategy
        self._initialize_weights(init_strategy, load_weights, weights_dir)

        if verbose:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[ConfigurableTextEncoder] Adapter: {BART_BASE_HIDDEN_DIM} -> {semantic_dim}")
            print(f"[ConfigurableTextEncoder] Config: h{self.adapter_config.hidden_size}_l{self.adapter_config.hidden_layers}")
            print(f"[ConfigurableTextEncoder] Trainable params: {trainable:,}")

    def _initialize_weights(
        self,
        init_strategy: InitStrategy,
        load_weights: bool,
        weights_dir: str,
    ) -> None:
        """Initialize adapter weights based on strategy."""
        if init_strategy == InitStrategy.RANDOM:
            self.adapter.initialize_random()
            if self.verbose:
                print(f"[ConfigurableTextEncoder] Initialized with random weights")
        elif init_strategy == InitStrategy.PRETRAINED and load_weights:
            # Try to load from experiment-specific path first
            if self.adapter.load(device=DEVICE):
                if self.verbose:
                    print(f"[ConfigurableTextEncoder] Loaded weights from {self.adapter.weights_path}")
            else:
                # Try loading from default text encoder path
                default_path = Path(weights_dir) / "facebook_bart-base_text_enc_weights.pth"
                if default_path.exists():
                    try:
                        weights = torch.load(default_path, map_location=DEVICE)
                        # Only load if architectures match
                        if self._check_weights_compatible(weights):
                            self.adapter.model.load_state_dict(weights)
                            if self.verbose:
                                print(f"[ConfigurableTextEncoder] Loaded default text encoder weights")
                        else:
                            if self.verbose:
                                print(f"[ConfigurableTextEncoder] Default weights incompatible, using random init")
                            self.adapter.initialize_random()
                    except Exception as e:
                        if self.verbose:
                            print(f"[ConfigurableTextEncoder] Could not load weights: {e}")
                        self.adapter.initialize_random()
                else:
                    if self.verbose:
                        print(f"[ConfigurableTextEncoder] No pretrained weights found, using random init")
                    self.adapter.initialize_random()

    def _check_weights_compatible(self, state_dict: dict) -> bool:
        """Check if loaded weights are compatible with current architecture."""
        try:
            self.adapter.model.load_state_dict(state_dict, strict=True)
            return True
        except RuntimeError:
            return False

    def forward(
        self,
        input_text: Union[str, list],
        *,
        max_length: Optional[int] = None,
        device: str = DEVICE,
    ) -> torch.Tensor:
        """
        Forward pass: tokenize -> encode with BART -> pool -> adapter

        Args:
            input_text: String or list of strings
            max_length: Maximum sequence length (optional override)
            device: Device to use

        Returns:
            Semantic vector representation (batch_size, semantic_dim)
        """
        if max_length is None:
            max_length = self.max_length

        # Move to device
        self.encoder = self.encoder.to(device)
        self.adapter = self.adapter.to(device)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(device)

        # Encode (no gradient through frozen encoder)
        with torch.no_grad():
            encoder_output = self.encoder(**inputs).last_hidden_state

        # Mean pool over sequence
        pooled_output = encoder_output.mean(dim=1)

        # Project through adapter
        semantic_vector = self.adapter(pooled_output)

        return semantic_vector


class ConfigurableAudioEncoder(nn.Module):
    """
    Configurable audio encoder using Whisper + custom MLP Adapter.
    """

    def __init__(
        self,
        semantic_dim: int = 1024,
        adapter_config: Optional[AdapterConfig] = None,
        base_model: str = "openai/whisper-small",
        weights_dir: str = "OptimalWeights",
        experiment_id: str = "",
        init_strategy: InitStrategy = InitStrategy.PRETRAINED,
        load_weights: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.semantic_dim = semantic_dim
        self.adapter_config = adapter_config or AdapterConfig()
        self.base_model_name = base_model
        self.verbose = verbose

        if verbose:
            print(f"[ConfigurableAudioEncoder] Loading Whisper from {base_model}...")

        self.processor = WhisperProcessor.from_pretrained(base_model)
        model = WhisperForConditionalGeneration.from_pretrained(base_model)
        self.encoder = model.get_encoder()

        # Freeze pretrained encoder
        self.encoder.requires_grad_(False)

        # Create configurable adapter
        prefix = f"{base_model.replace('/', '_')}_audio_enc"
        if experiment_id:
            prefix = f"{prefix}_{experiment_id}"

        self.adapter = ConfigurableAdapter(
            prefix=prefix,
            input_length=WHISPER_SMALL_ENCODER_DIM,
            output_length=semantic_dim,
            config=self.adapter_config,
            weights_dir=weights_dir,
        )

        self._initialize_weights(init_strategy, load_weights, weights_dir)

        if verbose:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[ConfigurableAudioEncoder] Adapter: {WHISPER_SMALL_ENCODER_DIM} -> {semantic_dim}")
            print(f"[ConfigurableAudioEncoder] Config: h{self.adapter_config.hidden_size}_l{self.adapter_config.hidden_layers}")
            print(f"[ConfigurableAudioEncoder] Trainable params: {trainable:,}")

    def _initialize_weights(
        self,
        init_strategy: InitStrategy,
        load_weights: bool,
        weights_dir: str,
    ) -> None:
        """Initialize adapter weights based on strategy."""
        if init_strategy == InitStrategy.RANDOM:
            self.adapter.initialize_random()
            if self.verbose:
                print(f"[ConfigurableAudioEncoder] Initialized with random weights")
        elif init_strategy == InitStrategy.PRETRAINED and load_weights:
            if self.adapter.load(device=DEVICE):
                if self.verbose:
                    print(f"[ConfigurableAudioEncoder] Loaded weights from {self.adapter.weights_path}")
            else:
                default_path = Path(weights_dir) / "openai_whisper-small_audio_enc_weights.pth"
                if default_path.exists():
                    try:
                        weights = torch.load(default_path, map_location=DEVICE)
                        self.adapter.model.load_state_dict(weights, strict=True)
                        if self.verbose:
                            print(f"[ConfigurableAudioEncoder] Loaded default audio encoder weights")
                    except Exception:
                        if self.verbose:
                            print(f"[ConfigurableAudioEncoder] Weights incompatible, using random init")
                        self.adapter.initialize_random()
                else:
                    if self.verbose:
                        print(f"[ConfigurableAudioEncoder] No pretrained weights found, using random init")
                    self.adapter.initialize_random()

    def forward(
        self,
        input_audio: Union[torch.Tensor, list],
        *,
        sampling_rate: int = 16000,
        device: str = DEVICE,
    ) -> torch.Tensor:
        """
        Forward pass: audio -> Whisper encoder -> pool -> adapter

        Args:
            input_audio: Raw audio waveform(s)
            sampling_rate: Audio sampling rate in Hz
            device: Device to use

        Returns:
            Semantic vector representation (batch_size, semantic_dim)
        """
        self.encoder = self.encoder.to(device)
        self.adapter = self.adapter.to(device)

        # Process audio to mel spectrogram
        input_features = self.processor(
            input_audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features.to(device)

        # Encode (no gradient through frozen encoder)
        with torch.no_grad():
            encoder_outputs = self.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state

        # Mean pool over time
        pooled_output = hidden_states.mean(dim=1)

        # Project through adapter
        semantic_vector = self.adapter(pooled_output)

        return semantic_vector


class ConfigurableImageEncoder(nn.Module):
    """
    Configurable image encoder using ViT + custom MLP Adapter.
    """

    def __init__(
        self,
        semantic_dim: int = 1024,
        adapter_config: Optional[AdapterConfig] = None,
        base_model: str = "nlpconnect/vit-gpt2-image-captioning",
        weights_dir: str = "OptimalWeights",
        experiment_id: str = "",
        init_strategy: InitStrategy = InitStrategy.PRETRAINED,
        load_weights: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.semantic_dim = semantic_dim
        self.adapter_config = adapter_config or AdapterConfig()
        self.base_model_name = base_model
        self.verbose = verbose

        if verbose:
            print(f"[ConfigurableImageEncoder] Loading ViT from {base_model}...")

        self.processor = ViTImageProcessor.from_pretrained(base_model)
        self.model = VisionEncoderDecoderModel.from_pretrained(base_model)

        # Freeze pretrained encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Create configurable adapter
        prefix = f"{base_model.replace('/', '_')}_image_enc"
        if experiment_id:
            prefix = f"{prefix}_{experiment_id}"

        self.adapter = ConfigurableAdapter(
            prefix=prefix,
            input_length=VIT_BASE_HIDDEN_DIM,
            output_length=semantic_dim,
            config=self.adapter_config,
            weights_dir=weights_dir,
        )

        self._initialize_weights(init_strategy, load_weights, weights_dir)

        if verbose:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[ConfigurableImageEncoder] Adapter: {VIT_BASE_HIDDEN_DIM} -> {semantic_dim}")
            print(f"[ConfigurableImageEncoder] Config: h{self.adapter_config.hidden_size}_l{self.adapter_config.hidden_layers}")
            print(f"[ConfigurableImageEncoder] Trainable params: {trainable:,}")

    def _initialize_weights(
        self,
        init_strategy: InitStrategy,
        load_weights: bool,
        weights_dir: str,
    ) -> None:
        """Initialize adapter weights based on strategy."""
        if init_strategy == InitStrategy.RANDOM:
            self.adapter.initialize_random()
            if self.verbose:
                print(f"[ConfigurableImageEncoder] Initialized with random weights")
        elif init_strategy == InitStrategy.PRETRAINED and load_weights:
            if self.adapter.load(device=DEVICE):
                if self.verbose:
                    print(f"[ConfigurableImageEncoder] Loaded weights from {self.adapter.weights_path}")
            else:
                default_path = Path(weights_dir) / "nlpconnect_vit-gpt2-image-captioning_image_enc_weights.pth"
                if default_path.exists():
                    try:
                        weights = torch.load(default_path, map_location=DEVICE)
                        self.adapter.model.load_state_dict(weights, strict=True)
                        if self.verbose:
                            print(f"[ConfigurableImageEncoder] Loaded default image encoder weights")
                    except Exception:
                        if self.verbose:
                            print(f"[ConfigurableImageEncoder] Weights incompatible, using random init")
                        self.adapter.initialize_random()
                else:
                    if self.verbose:
                        print(f"[ConfigurableImageEncoder] No pretrained weights found, using random init")
                    self.adapter.initialize_random()

    def forward(
        self,
        input_img: Union[Image, list],
        *,
        device: str = DEVICE,
    ) -> torch.Tensor:
        """
        Forward pass: image -> ViT encoder -> pool -> adapter

        Args:
            input_img: PIL Image or list of PIL Images
            device: Device to use

        Returns:
            Semantic vector representation (batch_size, semantic_dim)
        """
        self.model = self.model.to(device)
        self.adapter = self.adapter.to(device)

        # Process image
        pixel_values = self.processor(input_img, return_tensors="pt").pixel_values.to(device)

        # Encode (no gradient through frozen encoder)
        with torch.no_grad():
            encoder_outputs = self.model.encoder(pixel_values)
            hidden_states = encoder_outputs.last_hidden_state

        # Mean pool over patches
        pooled_output = hidden_states.mean(dim=1)

        # Project through adapter
        semantic_vector = self.adapter(pooled_output)

        return semantic_vector
