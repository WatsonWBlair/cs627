import torch
import numpy as np
import logging
from typing import Tuple, Optional

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


MODEL_NAME = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BART():
    def __init__(self) -> None:
        try:
            logging.info(f"Loading BART model '{MODEL_NAME}' to device: {DEVICE}")
            self.tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
            self.model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        except Exception as e:
            logging.error(f"Error loading model: {e}") # Exit gracefully if model load fails
            raise

    def encode(self, text_sentence: str, *, max_length: int = 1024, device: Optional[str] = DEVICE) -> Tuple[BaseModelOutput, torch.Tensor]:
            """
            Encodes an input sentence into the BART encoder and returns the full encoder context.

            Args:
                text_sentence (str):
                    Input English text that will be encoded by the BART encoder.
                max_length (int, optional):
                    Maximum token length for the tokenizer. Defaults to 1024.
                device (str, optional):
                    Device to run the model on ("cpu" or "cuda").
                    If None, uses global DEVICE.
            Returns:
                Tuple[BaseModelOutput, torch.Tensor]:
                    encoder_output:
                        The full hidden-state sequence from the encoder.
                    attention_mask:
                        Attention mask corresponding to the input sequence.
            Notes:
                - If the user passes an empty string, a zero placeholder encoder
                output is returned to avoid runtime crashes.
                - This function only performs encoding; no pooling or decoding.
            """

            if not text_sentence:
                logging.warning("Input sentence is empty. Returning zero tensors.")
                empty = BaseModelOutput(
                    last_hidden_state=torch.zeros((1, 1, self.model.config.hidden_size), device=device)
                )
                return empty, torch.zeros((1, 1), device=device)

            try:
                inputs = self.tokenizer(text_sentence, return_tensors="pt", max_length=max_length, truncation=True).to(device)
                with torch.no_grad():
                    encoder_output = self.model.model.encoder(**inputs)

                return encoder_output, inputs["attention_mask"]

            except Exception as e:
                logging.error(f"Error during encoding: {e}")
                empty = BaseModelOutput(
                    last_hidden_state=torch.zeros((1, 1, self.model.config.hidden_size), device=device)
                )
                return empty, torch.zeros((1, 1), device=device)

    def calculate_fixed_vector(self, encoder_output: BaseModelOutput, *, pooling: str = "mean") -> torch.Tensor:
            """
            Converts the encoder's hidden-state sequence into a fixed-size vector.

            Args:
                encoder_output (BaseModelOutput):
                    The full sequence output from the BART encoder.

                pooling (str, optional):
                    How to compress the sequence into one vector:
                        "mean" → mean pool across sequence length.
                        "cls"  → use token at position 0.
                    Default: "mean"

            Returns:
                torch.Tensor:
                    A 768-dimensional latent vector representing the entire input sentence.
            """
            if pooling == "cls":
                return encoder_output.last_hidden_state[:, 0, :]

            return torch.mean(encoder_output.last_hidden_state, dim=1)
