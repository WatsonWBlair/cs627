import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
from typing import Union
from src.utils.Adapter import Adapter

BASE_MODEL: str = "facebook/bart-base"
BART_BASE_HIDDEN_DIM: int = 768  # BART-base hidden dimension
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Vec_to_Text(torch.nn.Module):
    """
    Text decoder using Adapter + BART decoder.

    Current Implementation:
        - Supports single semantic vectors (batch, 1024)
        - Maps to BART encoder space via MLP adapter
        - Uses BART decoder for text generation

    Future Goals:
        - Variable-length sequence support (batch, seq_len, 1024)
        - Streaming generation capabilities
        - See: https://github.com/WatsonWBlair/cs627/issues for roadmap

    Args:
        base_model: HuggingFace BART model identifier (default: "facebook/bart-base")
        input_dim: Semantic vector dimension (default: 1024)
    """

    def __init__(self, base_model: str = BASE_MODEL, input_dim: int = 1024) -> None:
        super(Vec_to_Text, self).__init__()

        # Tokenizer for decoding text
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(base_model)

        # Pretrained BART decoder
        self.decoder: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(base_model)

        # Adapter: translates FROM semantic space (1024) TO BART embedding space (768)
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_dec",
            input_length=input_dim,  # Semantic vector size (1024)
            output_length=BART_BASE_HIDDEN_DIM,  # BART embedding size (768)
            hidden_size=200,
            hidden_layers=2
        )

        print(f"[Vec_to_Text] Decoder initialized with {base_model}")

        # Load trained adapter weights if they exist
        try:
            self.adapter.load(device=DEVICE)
            print(f"[Vec_to_Text] [OK] Loaded adapter weights: {self.adapter.weights_path}")
        except FileNotFoundError:
            print(f"[Vec_to_Text] [WARNING] No trained adapter weights found!")
            print(f"[Vec_to_Text] [WARNING] Expected: {self.adapter.weights_path}")
            print(f"[Vec_to_Text] [WARNING] Decoder will NOT work until trained.")
            print(f"[Vec_to_Text] [WARNING] Train using: python src/Training/train_adapters.py --mode decoder")

    def forward(self, semantic_vector: torch.Tensor, *, device: str = DEVICE) -> Union[str, list[str]]:
        """
        Generate text from semantic vector.

        Args:
            semantic_vector: Tensor of shape (batch_size, output_dim)
            device: Device to run on

        Returns:
            Generated text as string or list of strings
        """
        # Ensure input is on correct device
        semantic_vector = semantic_vector.to(device)
        self.adapter = self.adapter.to(device)
        self.decoder = self.decoder.to(device)

        # Project semantic vector to BART hidden space (1024 → 768)
        decoder_input = self.adapter(semantic_vector)  # Shape: (batch, 768)

        # Create encoder_outputs structure for BART decoder cross-attention
        encoder_outputs = BaseModelOutput(
            last_hidden_state=decoder_input.unsqueeze(1),  # (batch, 1, 768)
            hidden_states=None,
            attentions=None
        )

        # Generate text using encoder outputs
        output_ids = self.decoder.generate(
            encoder_outputs=encoder_outputs,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )

        # Decode to text
        text_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Return single string if batch size is 1
        if len(text_output) == 1:
            return text_output[0]
        return text_output


