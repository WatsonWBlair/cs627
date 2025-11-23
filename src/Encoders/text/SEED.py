import torch
import numpy as np
import logging
from typing import Tuple, Optional

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# This model is used as the seed of the system. All other encoder modules will train 
# using the `facebook/bart-base` encoding layer as gorund truth for encoding targets.

MODEL_NAME = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SEED():
    def __init__(self) -> None:
        super(SEED, self).__init__()
        try:
            logging.info(f"Loading BART model '{MODEL_NAME}' to device: {DEVICE}")
            self.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
            self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        except Exception as e:
            logging.error(f"Error loading model: {e}") # Exit gracefully if model load fails
            raise

        """
            Encodes an input sentence into the BART encoder and returns the full encoder context.

            Args:
            Returns:
            Notes:
        
        """
    def forward(self, input_text: str, *, max_length: int = 1024, device: Optional[str] = DEVICE) -> Tuple[BaseModelOutput, torch.Tensor]:
            if not input_text:
                logging.warning("Input sentence is empty. Returning zero tensors.")
                empty = BaseModelOutput(
                    last_hidden_state=torch.zeros((1, 1, self.model.config.hidden_size), device=device)
                )
                return empty, torch.zeros((1, 1), device=device)

            try:
                inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
                with torch.no_grad():
                    encoder_output = self.model.model.encoder(**inputs)

                return torch.mean(encoder_output.last_hidden_state, dim=1)
            
            except Exception as e:
                logging.error(f"Error during encoding: {e}")
                empty = BaseModelOutput(
                    last_hidden_state=torch.zeros((1, 1, self.model.config.hidden_size), device=device)
                )
                return empty

    def batch(self, inputs: list[str]):
        results = [self.forward(input) for input in inputs]
        return results