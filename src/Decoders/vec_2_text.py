import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from typing import Union
from utils.Adapter import Adapter

BASE_MODEL: str = "facebook/bart-base"
BART_BASE_HIDDEN_DIM: int = 768  # BART-base hidden dimension
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Vec_to_Text(torch.nn.Module):
    """
    Text decoder using Adapter + BART decoder.

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

        # Project semantic vector to decoder input space
        decoder_input = self.adapter(semantic_vector)

        # Generate text
        # Note: This is simplified; proper implementation needs encoder_outputs formatting
        output_ids = self.decoder.generate(
            inputs_embeds=decoder_input.unsqueeze(1),  # Add sequence dimension
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


