import torch
from transformers import  ViTImageProcessor, VisionEncoderDecoderModel

from utils.Adapter import Adapter

BASE_MODEL = "nlpconnect/vit-gpt2-image-captioning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Image_to_Vec(torch.nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, token_length: int = 1024) -> None:
        super(Image_to_Vec, self).__init__()
        adapterModel = Adapter(token_length==token_length, hidden_size=200,hidden_layers=2)
        self.model = VisionEncoderDecoderModel.from_pretrained(base_model, adapterModel)
        self.tokenizer = ViTImageProcessor.from_pretrained(base_model, max_length=token_length) #image_processor

       
    def forward(self, input_img, *, max_length: int = 1024, device: str = DEVICE):
        pixel_values = self.tokenizer(input_img, return_tensors="pt", max_length=max_length).to(device).pixel_values
        output = self.model.generate(pixel_values, skip_special_tokens=True)[0]

        return output


    def tokenize(self, input_img):
        return self.tokenizer(input_img, return_tensors="pt", max_length=1024).to(DEVICE).pixel_values
        


# from transformers import Pipeline


# class Img_to_descripton(Pipeline):
#     def _sanitize_parameters(self, **kwargs):
#         "_sanitize_parameters exists to allow users to pass any parameters whenever they wish, be it at initialization time pipeline(...., maybe_arg=4) or at call time pipe = pipeline(...); output = pipe(...., maybe_arg=4)."
#         preprocess_kwargs = {}
#         if "maybe_arg" in kwargs:
#             preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
#         return preprocess_kwargs, {}, {}

#     def preprocess(self, inputs, maybe_arg=2):
#         """
#         will take the originally defined inputs, and turn them into something feedable to the model. It might contain more information and is usually a Dict.
#         """
#         model_input = Tensor(inputs["input_ids"])
#         return {"model_input": model_input}

#     def _forward(self, model_inputs):
#         """
#         _forward is the implementation detail and is not meant to be called directly. forward is the preferred called method as it contains safeguards to make sure everything is working on the expected device. If anything is linked to a real model it belongs in the _forward method, anything else is in the preprocess/postprocess.
#         """
#         # model_inputs == {"model_input": model_input}
#         outputs = self.model(**model_inputs)
#         # Maybe {"logits": Tensor(...)}
#         return outputs

#     def postprocess(self, model_outputs):
#         """
#         postprocess methods will take the output of _forward and turn it into the final output that was decided earlier.
#         """
#         best_class = model_outputs["logits"].softmax(-1)
#         return best_class