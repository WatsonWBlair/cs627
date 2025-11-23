import torch
from transformers import EncoderDecoderModel, BertTokenizer
from utils.Adapter import Adapter

BASE_MODEL = "openai-community/gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SEED():
    def __init__(self, pretrained: str, max_length: int = 1024) -> None:
        super(SEED, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BASE_MODEL, max_length=max_length)
        adapterModel = Adapter(max_length,200,max_length) # TODO: Load from checkpoint using pretrained arg
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(BASE_MODEL, adapterModel)

       
    def forward(self, input_text: str, *, max_length: int = 1024, device: str = DEVICE):
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=max_length).to(device).input_ids
        output = self.model.generate(inputs)[0]

        return output


