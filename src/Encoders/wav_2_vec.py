import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from utils.Adapter import Adapter

BASE_MODEL = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Audio_to_Vec(torch.nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, token_length: int = 1024) -> None:
        super(Audio_to_Vec, self).__init__()

        self.processor = WhisperProcessor.from_pretrained(base_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(base_model)
        self.model.config.forced_decoder_ids = None

        # Create adapter with proper parameters
        self.adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_audio_enc",
            input_length=token_length,
            output_length=token_length,
            hidden_size=200,
            hidden_layers=2
        )

    def forward(self, input_audio, *, sampling_rate: int = 16000, device: str = DEVICE):
        # Process audio input
        input_features = self.processor(
            input_audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(device)

        # Generate speech-to-text tokens
        stt_tokens = self.model.to(device).generate(input_features)

        # Convert to semantic vector via adapter
        semantic_vector = self.adapter(stt_tokens)

        return semantic_vector


