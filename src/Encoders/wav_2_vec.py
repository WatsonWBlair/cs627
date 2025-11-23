import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from utils.Adapter import Adapter

BASE_MODEL = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SEED():
    def __init__(self, max_length: int = 1024) -> None:
        self.tokenizer = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.adaptor = Adapter()
        self.model.config.forced_decoder_ids = None
        # TODO: Implement audio input of conversational speech
        # super(SEED, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained(BASE_MODEL, max_length=max_length)
        # adapterModel = Adapter(max_length,200,max_length) # TODO: Load from checkpoint using pretrained arg
        # self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(BASE_MODEL, adapterModel)

       
    def forward(self, input_chunk: torch.Tensor, *, device: str = DEVICE):
        input_tokens = self.tokenizer(input_chunk, sampling_rate=input_chunk["sampling_rate"], return_tensors="pt").input_features
        stt_tokens = self.model.to(device).generate(input_tokens)
        
        semantic_vector = self.adaptor(stt_tokens)

        return semantic_vector


