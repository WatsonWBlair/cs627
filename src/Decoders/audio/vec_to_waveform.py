"""
Vec_to_Audio: Semantic Vector to Speech Decoder using SpeechT5

Architecture: Semantic Vector (1024) -> Adapter (768) -> SpeechT5 -> HiFi-GAN -> Waveform

Status: EXPERIMENTAL
- Direct embedding injection approach
- Affective speech learned end-to-end from semantic content
- Training pipeline implemented in src/Training/train_audio_decoder.py

Based on: microsoft/speecht5_tts with HiFi-GAN vocoder
"""

import torch
import numpy as np
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    SpeechT5HifiGan
)
from datasets import load_dataset
from typing import Union, Optional, Tuple
from src.utils.Adapter import Adapter

# Model identifiers
TTS_MODEL: str = "microsoft/speecht5_tts"
VOCODER_MODEL: str = "microsoft/speecht5_hifigan"

# Audio specifications (matching Audio_to_Vec encoder)
SAMPLE_RATE: int = 16000

# SpeechT5 hidden dimension
SPEECHT5_HIDDEN_DIM: int = 768

# Semantic vector dimension
SEMANTIC_DIM: int = 1024

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Vec_to_Audio(torch.nn.Module):
    """
    Audio decoder using Adapter + SpeechT5 + HiFi-GAN.

    Architecture: Semantic Vector (1024) -> Adapter (768) -> SpeechT5 Decoder -> HiFi-GAN -> Waveform

    Affective Speech: Prosody is learned end-to-end from semantic content.
    The adapter learns to map emotion-encoded semantic vectors to appropriate
    hidden representations that influence SpeechT5's prosody generation.

    Args:
        tts_model: SpeechT5 TTS model identifier
        vocoder_model: HiFi-GAN vocoder identifier
        input_dim: Semantic vector dimension (default: 1024)
        freeze_tts: Whether to freeze SpeechT5 weights (default: True)
        default_speaker_idx: Index in CMU ARCTIC x-vectors dataset (default: 7306)
    """

    def __init__(
        self,
        tts_model: str = TTS_MODEL,
        vocoder_model: str = VOCODER_MODEL,
        input_dim: int = SEMANTIC_DIM,
        freeze_tts: bool = True,
        default_speaker_idx: int = 7306  # Female speaker from CMU ARCTIC
    ) -> None:
        super(Vec_to_Audio, self).__init__()

        # Store config
        self.tts_model_name = tts_model
        self.vocoder_model_name = vocoder_model
        self.sample_rate = SAMPLE_RATE

        # Adapter: Semantic space (1024) -> SpeechT5 hidden space (768)
        self.adapter: Adapter = Adapter(
            prefix=f"{tts_model.replace('/', '_')}_audio_dec",
            input_length=input_dim,
            output_length=SPEECHT5_HIDDEN_DIM,
            hidden_size=200,
            hidden_layers=2
        )

        # SpeechT5 TTS model
        self.processor: SpeechT5Processor = SpeechT5Processor.from_pretrained(tts_model)
        self.tts_model: SpeechT5ForTextToSpeech = SpeechT5ForTextToSpeech.from_pretrained(tts_model)

        # HiFi-GAN vocoder for mel -> waveform
        self.vocoder: SpeechT5HifiGan = SpeechT5HifiGan.from_pretrained(vocoder_model)

        # Freeze pretrained models (only adapter is trainable)
        if freeze_tts:
            for param in self.tts_model.parameters():
                param.requires_grad = False
            for param in self.vocoder.parameters():
                param.requires_grad = False
            print(f"[Vec_to_Audio] SpeechT5 and HiFi-GAN frozen")

        # Load default speaker embedding from CMU ARCTIC x-vectors
        self._load_speaker_embedding(default_speaker_idx)

        # Track trainable parameters
        trainable = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"[Vec_to_Audio] Decoder initialized with {tts_model}")
        print(f"[Vec_to_Audio] Adapter has {trainable:,} trainable parameters")

        # Try to load trained adapter weights
        try:
            self.adapter.load(device=DEVICE)
            print(f"[Vec_to_Audio] [OK] Loaded adapter weights: {self.adapter.weights_path}")
        except FileNotFoundError:
            print(f"[Vec_to_Audio] [WARNING] No trained adapter weights found!")
            print(f"[Vec_to_Audio] [WARNING] Expected: {self.adapter.weights_path}")
            print(f"[Vec_to_Audio] [WARNING] Decoder will NOT work until trained.")
            print(f"[Vec_to_Audio] [WARNING] Train using: python src/Training/train_audio_decoder.py")

    def _load_speaker_embedding(self, speaker_idx: int) -> None:
        """Load pre-computed speaker embedding from CMU ARCTIC dataset."""
        try:
            xvectors = load_dataset(
                "Matthijs/cmu-arctic-xvectors",
                split="validation"
            )
            self.speaker_embedding = torch.tensor(
                xvectors[speaker_idx]["xvector"]
            ).unsqueeze(0)  # (1, 512)
            print(f"[Vec_to_Audio] Loaded speaker embedding (idx={speaker_idx})")
        except Exception as e:
            # Fallback: random speaker embedding (won't sound natural but allows testing)
            print(f"[Vec_to_Audio] [WARNING] Could not load speaker embedding: {e}")
            print(f"[Vec_to_Audio] [WARNING] Using random speaker embedding (for testing only)")
            self.speaker_embedding = torch.randn(1, 512)

    def forward(
        self,
        semantic_vector: torch.Tensor,
        *,
        device: str = DEVICE,
        speaker_embedding: Optional[torch.Tensor] = None,
        max_length: int = 1000
    ) -> np.ndarray:
        """
        Generate audio waveform from semantic vector.

        Args:
            semantic_vector: Tensor of shape (batch_size, 1024)
            device: Device to run on
            speaker_embedding: Optional custom speaker embedding (batch, 512)
            max_length: Maximum output length in mel frames

        Returns:
            Audio waveform as numpy array (samples,) at 16kHz

        Note: Currently optimized for batch_size=1
        """
        # Move to device
        semantic_vector = semantic_vector.to(device)
        self.adapter = self.adapter.to(device)
        self.tts_model = self.tts_model.to(device)
        self.vocoder = self.vocoder.to(device)

        # Use default or custom speaker embedding
        if speaker_embedding is None:
            spk_emb = self.speaker_embedding.to(device)
        else:
            spk_emb = speaker_embedding.to(device)

        # Ensure correct batch size for speaker embedding
        batch_size = semantic_vector.shape[0]
        if spk_emb.shape[0] != batch_size:
            spk_emb = spk_emb.expand(batch_size, -1)

        # 1. Adapter: semantic vector (1024) -> hidden (768)
        hidden_states = self.adapter(semantic_vector)  # (batch, 768)

        # 2. Generate mel spectrogram using SpeechT5 decoder
        # Note: SpeechT5's generate_speech doesn't accept encoder_outputs directly,
        # so we use a custom generation approach with the internal decoder
        with torch.no_grad():
            spectrogram = self._generate_mel_from_hidden(
                hidden_states=hidden_states.unsqueeze(1),  # (batch, 1, 768)
                speaker_embeddings=spk_emb,
                max_length=max_length
            )

        # 3. Apply HiFi-GAN vocoder: mel -> waveform
        with torch.no_grad():
            waveform = self.vocoder(spectrogram)

        # Convert to numpy (16kHz, mono)
        audio = waveform.cpu().numpy().squeeze()

        return audio

    def generate_with_mel(
        self,
        semantic_vector: torch.Tensor,
        *,
        device: str = DEVICE,
        speaker_embedding: Optional[torch.Tensor] = None,
        max_length: int = 1000
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Generate audio with intermediate mel spectrogram (for training/analysis).

        Args:
            semantic_vector: Tensor of shape (batch_size, 1024)
            device: Device to run on
            speaker_embedding: Optional custom speaker embedding (batch, 512)
            max_length: Maximum output length in mel frames

        Returns:
            Tuple of (waveform as numpy array, mel spectrogram as tensor)
        """
        # Move to device
        semantic_vector = semantic_vector.to(device)
        self.adapter = self.adapter.to(device)
        self.tts_model = self.tts_model.to(device)
        self.vocoder = self.vocoder.to(device)

        # Use default or custom speaker embedding
        if speaker_embedding is None:
            spk_emb = self.speaker_embedding.to(device)
        else:
            spk_emb = speaker_embedding.to(device)

        # Ensure correct batch size for speaker embedding
        batch_size = semantic_vector.shape[0]
        if spk_emb.shape[0] != batch_size:
            spk_emb = spk_emb.expand(batch_size, -1)

        # 1. Adapter: semantic vector (1024) -> hidden (768)
        hidden_states = self.adapter(semantic_vector)  # (batch, 768)

        # 2. Generate mel spectrogram using custom decoder path
        with torch.no_grad():
            spectrogram = self._generate_mel_from_hidden(
                hidden_states=hidden_states.unsqueeze(1),  # (batch, 1, 768)
                speaker_embeddings=spk_emb,
                max_length=max_length
            )

        # 3. Apply vocoder
        with torch.no_grad():
            waveform = self.vocoder(spectrogram)

        # Convert waveform to numpy
        audio = waveform.cpu().numpy().squeeze()

        return audio, spectrogram

    def get_adapter_output(
        self,
        semantic_vector: torch.Tensor,
        *,
        device: str = DEVICE
    ) -> torch.Tensor:
        """
        Get adapter output for training (differentiable path).

        This method is used during training to get the adapter output
        before the non-differentiable TTS generation step.

        Args:
            semantic_vector: Tensor of shape (batch_size, 1024)
            device: Device to run on

        Returns:
            Hidden states tensor of shape (batch_size, 768)
        """
        semantic_vector = semantic_vector.to(device)
        self.adapter = self.adapter.to(device)

        # Adapter forward pass (differentiable)
        hidden_states = self.adapter(semantic_vector)

        return hidden_states

    def _generate_mel_from_hidden(
        self,
        hidden_states: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        max_length: int = 1000,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0
    ) -> torch.Tensor:
        """
        Generate mel spectrogram from encoder hidden states.

        This bypasses the text encoder and directly uses the speech decoder
        with our adapter-generated hidden states.

        Adapted from HuggingFace SpeechT5 generate_speech implementation.

        Args:
            hidden_states: (batch, seq_len, 768) encoder hidden states from adapter
            speaker_embeddings: (batch, 512) speaker x-vectors
            max_length: Maximum mel spectrogram length
            threshold: Stop threshold for autoregressive generation
            minlenratio: Minimum length ratio
            maxlenratio: Maximum length ratio

        Returns:
            Mel spectrogram tensor (seq_len, num_mel_bins)
        """
        # Get model components
        model = self.tts_model

        # Set up encoder outputs format expected by decoder
        encoder_attention_mask = torch.ones(
            hidden_states.shape[:2],
            dtype=torch.long,
            device=hidden_states.device
        )

        # Get batch size (currently optimized for batch=1)
        batch_size = hidden_states.shape[0]

        # Compute min/max lengths
        encoder_seq_len = hidden_states.shape[1]
        minlen = int(encoder_seq_len * minlenratio)
        maxlen = int(encoder_seq_len * maxlenratio)

        # Initialize autoregressive generation
        # SpeechT5 speech decoder expects: (batch, num_mel_bins) as initial input
        num_mel_bins = model.config.num_mel_bins  # Usually 80

        # Start with zeros (silence) as initial decoder input
        decoder_input = torch.zeros(
            (batch_size, 1, num_mel_bins),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # Initialize output collection
        all_mel_outputs = []
        past_key_values = None

        for step in range(max_length):
            # Get the last frame as input to decoder
            if step == 0:
                current_input = decoder_input
            else:
                current_input = all_mel_outputs[-1].unsqueeze(1)

            # Apply speech decoder prenet
            prenet_output = model.speecht5.decoder.prenet(current_input, speaker_embeddings)

            # Decoder forward pass
            decoder_outputs = model.speecht5.decoder.wrapped_decoder(
                hidden_states=prenet_output,
                attention_mask=None,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            past_key_values = decoder_outputs.past_key_values
            decoder_hidden = decoder_outputs.last_hidden_state

            # Apply speech decoder postnet to get mel spectrogram
            # postnet output: (batch, 1, num_mel_bins)
            mel_output = model.speech_decoder_postnet.feat_out(decoder_hidden)
            mel_output = mel_output.squeeze(1)  # (batch, num_mel_bins)

            all_mel_outputs.append(mel_output)

            # Check stop condition using stop token prediction
            prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(decoder_hidden))

            if step >= minlen and (prob > threshold).all():
                break

        # Stack all mel frames: (seq_len, batch, num_mel_bins) -> (batch, seq_len, num_mel_bins)
        mel_spectrogram = torch.stack(all_mel_outputs, dim=1)

        # Apply postnet refinement if available
        if hasattr(model.speech_decoder_postnet, 'postnet'):
            # postnet expects (batch, num_mel_bins, seq_len)
            mel_transposed = mel_spectrogram.transpose(1, 2)
            mel_refined = model.speech_decoder_postnet.postnet(mel_transposed)
            mel_spectrogram = (mel_transposed + mel_refined).transpose(1, 2)

        # Return single item for batch=1 (matches vocoder expectations)
        if batch_size == 1:
            return mel_spectrogram.squeeze(0)  # (seq_len, num_mel_bins)

        return mel_spectrogram
