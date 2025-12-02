import torch
from transformers import ViTImageProcessor, VisionEncoderDecoderModel
from typing import Union
from utils.Adapter import Adapter

BASE_MODEL: str = "nlpconnect/vit-gpt2-image-captioning"
VIT_BASE_HIDDEN_DIM: int = 768  # ViT-base encoder output dimension
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Image_to_Vec(torch.nn.Module):
    """
    Image encoder using ViT + MLP Adapter.

    Args:
        base_model: HuggingFace ViT model identifier (default: "nlpconnect/vit-gpt2-image-captioning")
        output_dim: Semantic vector space dimension (default: 1024)
    """

    def __init__(self, base_model: str = BASE_MODEL, output_dim: int = 1024) -> None:
        super(Image_to_Vec, self).__init__()

        self.processor: ViTImageProcessor = ViTImageProcessor.from_pretrained(base_model)

        self.model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(base_model)

        # Create adapter: ViT hidden dim (768) → Semantic vector space (1024)
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_image_enc",
            input_length=VIT_BASE_HIDDEN_DIM,
            output_length=output_dim,
            hidden_size=200,
            hidden_layers=2
        )

    def forward(self, input_img: Union[object, list], *, device: str = DEVICE) -> torch.Tensor:
        # Process image input
        pixel_values = self.processor(input_img, return_tensors="pt").pixel_values.to(device)

        # Extract continuous embeddings from vision encoder (not discrete tokens)
        encoder_outputs = self.model.encoder(pixel_values)

        # Get last hidden state and apply mean pooling
        hidden_states = encoder_outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_dim)
        pooled_output = hidden_states.mean(dim=1)  # Shape: (batch, hidden_dim)

        # Convert to semantic vector via adapter
        semantic_vector = self.adapter(pooled_output.to(device))

        return semantic_vector
        


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