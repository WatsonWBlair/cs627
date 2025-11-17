import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
import logging

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

    def encode(self, text_sentence: str) -> torch.Tensor:
        """
        Encodes an English text sentence into a shared semantic latent space.

        In the context of sequence-to-sequence models like BART, the 'latent space'
        is derived from the final hidden states of the encoder. We use the
        mean-pooled sequence vector as the sentence embedding (latent vector).

        Args:
            text_sentence: The input English text string.

        Returns:
            A torch.Tensor representing the semantic latent vector (embedding).
            Shape: [1, hidden_size]
        """
        if not text_sentence:
            logging.warning("Input sentence is empty. Returning zero tensor.")
            return torch.zeros((1, self.model.config.hidden_size), device=DEVICE)

        try:
            inputs = self.tokenizer(text_sentence, return_tensors='pt', 
                            max_length=1024, truncation=True).to(DEVICE)

            with torch.no_grad():
                encoder_output = self.model.model.encoder(**inputs)

            last_hidden_state = encoder_output.last_hidden_state
            
            latent_vector = torch.mean(last_hidden_state, dim=1)
            
            return latent_vector

        except Exception as e:
            logging.error(f"Error during encoding: {e}")
            return torch.zeros((1, self.model.config.hidden_size), device=DEVICE)


