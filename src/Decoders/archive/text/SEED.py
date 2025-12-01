import torch
from transformers import EncoderDecoderModel, BertTokenizer

from utils.Adapter import Adapter

# This model is used as the seed of the system. All other encoder modules will train 
# using the `facebook/bart-base` encoding layer as gorund truth for encoding targets.

BASE_MODEL = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SEED():
    def __init__(self, use_saved: bool, base_model: str = BASE_MODEL, token_length: int = 1024) -> None:
        super(SEED, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(base_model, max_length=token_length)
        adapterModel = Adapter(input_size=token_length, hidden_size=200,output_size=token_length,hidden_layers=2)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(adapterModel,base_model)

    def forward(self, input_vector):
        decoded_vector = self.model.generate(input_vector)
        text_output = self.tokenizer.decode(decoded_vector, skip_special_tokens=True)

        return text_output

