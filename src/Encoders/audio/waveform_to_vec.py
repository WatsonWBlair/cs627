import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Union
from src.utils.Adapter import Adapter

BASE_MODEL: str = "openai/whisper-small"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_SMALL_ENCODER_DIM: int = 768


class Audio_to_Vec(torch.nn.Module):
    """
    Audio encoder using Whisper encoder + MLP Adapter.

    Args:
        base_model: Pretrained Whisper model name (default: "openai/whisper-small")
        output_dim: Output semantic vector dimension (default: 1024)
        freeze_encoder: Whether to freeze Whisper encoder (default: True)
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        output_dim: int = 1024,
        freeze_encoder: bool = True
    ) -> None:
        super(Audio_to_Vec, self).__init__()

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(base_model)
        self.model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(base_model)

        # Extract just the encoder for efficiency
        self.encoder = self.model.get_encoder()

        # Freeze pretrained encoder to prevent catastrophic forgetting
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            print(f"[Audio_to_Vec] Whisper encoder frozen (0 trainable params)")

        # Create adapter: maps encoder hidden states to semantic space
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_audio_enc",
            input_length=WHISPER_SMALL_ENCODER_DIM,
            output_length=output_dim,
            hidden_size=200,
            hidden_layers=2
        )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Audio_to_Vec] Adapter has {trainable_params:,} trainable parameters")

    def forward(
        self,
        input_audio: Union[torch.Tensor, list],
        *,
        sampling_rate: int = 16000,
        device: str = DEVICE
    ) -> torch.Tensor:
        """
        Forward pass: audio waveform → encoder hidden states → pooled → adapter → semantic vector

        Args:
            input_audio: Raw audio waveform(s)
                - Single: numpy array or tensor of shape (samples,)
                - Batch: list of arrays or tensor of shape (batch, samples)
            sampling_rate: Audio sampling rate in Hz (default: 16000)
            device: Device to run on

        Returns:
            Semantic vector representation (batch_size, output_dim)
        """
        # Move encoder to device
        self.encoder = self.encoder.to(device)

        # Process audio input → log-mel spectrogram features
        input_features = self.processor(
            input_audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(device)

        # Pass through Whisper encoder (no gradients if frozen)
        with torch.set_grad_enabled(self.encoder.training and any(p.requires_grad for p in self.encoder.parameters())):
            encoder_outputs = self.encoder(input_features)
            encoder_hidden_states = encoder_outputs.last_hidden_state  # (batch, time, 768)

        # Mean pooling over time dimension to get fixed-size representation
        pooled_output = encoder_hidden_states.mean(dim=1)  # (batch, 768)

        # Project to semantic space with adapter
        semantic_vector = self.adapter(pooled_output)

        return semantic_vector


