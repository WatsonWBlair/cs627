import torch
from transformers import EncoderDecoderModel, BertTokenizer, Pipeline, AutoTokenizer, AutoModelForSequenceClassification
from utils.Adapter import Adapter

# Text Decoder Pipelines: Semantic Vector to Text modality

# This model is used as the seed of the system. All other encoder modules will train 
# using the `facebook/bart-base` encoding layer as gorund truth for encoding targets.

BASE_MODEL = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Vec_to_Text(Pipeline):
    def __init__(self, base_model: str = BASE_MODEL, token_length: int = 1024) -> None:
        super(Vec_to_Text, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(base_model, max_length=token_length)
        adapterModel = Adapter(token_length=token_length, hidden_size=200,hidden_layers=2)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(adapterModel,base_model)

    def forward(self, input_vector):
        decoded_vector = self.model.generate(input_vector)
        text_output = self.tokenizer.decode(decoded_vector, skip_special_tokens=True)

        return text_output


