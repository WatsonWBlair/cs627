import torch
from transformers import EncoderDecoderModel, BertTokenizer
from transformers import Pipeline, AutoTokenizer, AutoModelForSequenceClassification

from utils.Adapter import Adapter

BASE_MODEL = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Text_to_Vec(torch.nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, token_length: int = 1024) -> None:
        super(Text_to_Vec, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(base_model, max_length=token_length)
        adapterModel = Adapter(token_length=token_length, hidden_size=200 ,hidden_layers=2)
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(base_model, adapterModel)


    def forward(self, input_text: str, device: str = DEVICE):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device).input_ids
        output = self.model.generate(inputs)[0]

        return output


# class TextPipeline(Pipeline):
#     def _sanitize_parameters(self, **kwargs):
#         # Example: Ensure a 'threshold' parameter is valid
#         if "threshold" in kwargs and not isinstance(kwargs["threshold"], (int, float)):
#             raise ValueError("Threshold must be a number.")
#         return {}, {}, {} # return model_kwargs, preprocess_params, postprocess_params

#     def preprocess(self, text):
#         tokenizer = AutoTokenizer.from_pretrained(self.model.name_or_path)
#         return tokenizer(text, return_tensors="pt")

#     def _forward(self, model_inputs):
#         return self.model(**model_inputs)

#     def postprocess(self, model_outputs, threshold=0.5):
#         logits = model_outputs.logits
#         probabilities = logits.softmax(dim=-1)
#         predictions = (probabilities[:, 1] > threshold).long() # Assuming binary classification
#         labels = ["negative", "positive"]
#         return [{"label": labels[p.item()], "score": prob.item()} for p, prob in zip(predictions, probabilities[:, 1])]

