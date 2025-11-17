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

def decode(latent_vector: torch.Tensor) -> str:
    decoded_text = self.tokenizer.decode(latent_vector, 
                                    skip_special_tokens=True)
    
    return decoded_text