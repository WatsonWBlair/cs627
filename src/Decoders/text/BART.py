import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import numpy as np
import logging
from typing import Tuple, Optional

MODEL_NAME = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BART():
    def __init__(self) -> None:
        try:
            logging.info(f"Loading BART model '{MODEL_NAME}' to device: {DEVICE}")
            self.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
            self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            # Exit gracefully if model load fails
            raise

def decode(
    self,
    encoder_output: BaseModelOutput,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int = 40,
    num_beams: int = 4,
    device: Optional[str] = DEVICE
) -> str:
    """
    Decodes text directly from a BART encoder output using model.generate().

    Args:
        encoder_output (BaseModelOutput):
            Encoder hidden-state sequence to be decoded.

        attention_mask (torch.Tensor):
            Attention mask for the original input sequence.

        max_new_tokens (int, optional):
            Maximum generated tokens. Defaults to 40.

        num_beams (int, optional):
            Number of beams for beam-search decoding. Defaults to 4.

        device (str, optional):
            Device for inference. Default uses global DEVICE.

    Returns:
        str:
            The decoded text generated from the latent sequence context.

    Notes:
        - The caller *must* ensure the encoder_output and attention_mask come
          from the same input batch.
        - This function **expects valid encoder_output**, not latent vectors.
    """

    try:
        generated_ids = self.model.generate(
            encoder_outputs=encoder_output,
            attention_mask=attention_mask,
            max_length=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

        decoded_text = self.tokenizer.decode(
            generated_ids.squeeze(),
            skip_special_tokens=True
        ).to(device)
        return decoded_text

    except Exception as e:
        logging.error(f"Error during decoding: {e}")
        return ""