import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from typing import Union
import numpy as np
from src.utils.Adapter import Adapter

BASE_MODEL: str = "microsoft/wavlm-base"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
WAVLM_BASE_DIM: int = 768


class Tone_to_Vec(torch.nn.Module):
    """
    Audio tone/prosody encoder using WavLM encoder + MLP Adapter.

    Status: PRODUCTION

    Specialized for emotional tone, prosody, and speaking style detection.
    Uses WavLM (optimized for tone) instead of Whisper (optimized for ASR).

    Args:
        base_model: Pretrained WavLM model name (default: "microsoft/wavlm-base")
        output_dim: Output semantic vector dimension (default: 1024)
        freeze_encoder: Whether to freeze WavLM encoder (default: True)
    """

    def __init__(
        self,
        base_model: str = BASE_MODEL,
        output_dim: int = 1024,
        freeze_encoder: bool = True
    ) -> None:
        super(Tone_to_Vec, self).__init__()

        self.feature_extractor: Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)
        self.model: WavLMModel = WavLMModel.from_pretrained(base_model)

        # Extract encoder for efficiency
        self.encoder = self.model

        # Freeze pretrained encoder to prevent catastrophic forgetting
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            print(f"[Tone_to_Vec] WavLM encoder frozen (0 trainable params)")

        # Create adapter: maps encoder hidden states to semantic space
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_audio_tone_enc",
            input_length=WAVLM_BASE_DIM,
            output_length=output_dim,
            hidden_size=200,
            hidden_layers=2
        )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Tone_to_Vec] Adapter has {trainable_params:,} trainable parameters")

    def forward(
        self,
        input_audio: Union[torch.Tensor, list, np.ndarray],
        *,
        sampling_rate: int = 16000,
        device: str = DEVICE,
        pregen: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: audio waveform → WavLM encoder → pooled → adapter → semantic vector

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

        # Process audio input → WavLM input features
        input_values = self.feature_extractor(
            input_audio,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt"
        ).input_values.to(device)

        # Pass through WavLM encoder (no gradients if frozen)
        with torch.set_grad_enabled(self.encoder.training and any(p.requires_grad for p in self.encoder.parameters())):
            encoder_outputs = self.encoder(input_values)
            encoder_hidden_states = encoder_outputs.last_hidden_state  # (batch, time, 768)

        # Mean pooling over time dimension to get fixed-size representation
        pooled_output = encoder_hidden_states.mean(dim=1)  # (batch, 768)
        
        # If in pregeneration mode
        if(pregen):
            return pooled_output
        
        # Project to semantic space with adapter
        semantic_vector = self.adapter(pooled_output)

        return semantic_vector


