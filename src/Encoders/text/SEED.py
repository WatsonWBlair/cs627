import torch
from transformers import EncoderDecoderModel, BertTokenizer
from utils.Adapter import Adapter

BASE_MODEL = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Text_to_Vec(torch.nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, token_length: int = 1024) -> None:
        super(Text_to_Vec, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(base_model, max_length=token_length)
        adapterModel = Adapter(input_size=token_length, hidden_size=200,output_size=token_length,hidden_layers=2)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(base_model, adapterModel)

       
    def forward(self, input_text: str, *, max_length: int = 1024, device: str = DEVICE):
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length).to(device).input_ids
        output = self.model.generate(inputs)[0]

        return output


