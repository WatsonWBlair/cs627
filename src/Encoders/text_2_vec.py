import torch
from transformers import BartModel, BartTokenizer
from typing import Union, Optional
from utils.Adapter import Adapter

BASE_MODEL: str = "facebook/bart-base"
BART_BASE_HIDDEN_DIM: int = 768  # BART-base hidden dimension
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Text_to_Vec(torch.nn.Module):
    """
    Text encoder using BART + MLP Adapter.

    Args:
        base_model: HuggingFace BART model identifier (default: "facebook/bart-base")
        max_length: Maximum sequence length (default: 1024)
        output_dim: Semantic vector space dimension (default: 1024)
    """

    def __init__(self, base_model: str = BASE_MODEL, max_length: int = 1024, output_dim: int = 1024) -> None:
        super(Text_to_Vec, self).__init__()
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(base_model)
        self.max_length: int = max_length

        # Load pretrained BART encoder
        self.encoder: BartModel = BartModel.from_pretrained(base_model)

        # Adapter: translates FROM BART embedding space (768) TO semantic space (1024)
        self.adapter: Adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_text_enc",
            input_length=BART_BASE_HIDDEN_DIM,  # BART hidden dim (768)
            output_length=output_dim,  # Semantic vector size (1024)
            hidden_size=200,
            hidden_layers=2
        )

    def forward(
        self,
        input_text: Union[str, list[str]],
        *,
        max_length: Optional[int] = None,
        device: str = DEVICE
    ) -> torch.Tensor:
        """
        Forward pass: tokenize -> encode with BART -> project with adapter

        Args:
            input_text: String or list of strings
            max_length: Maximum sequence length
            device: Device to use

        Returns:
            Semantic vector representation (batch_size, embed_dim)
        """
        if max_length is None:
            max_length = self.max_length

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)

        # Get encoder output (use last_hidden_state)
        encoder_output = self.encoder(**inputs).last_hidden_state

        # Pool encoder output (mean pooling over sequence)
        pooled_output = encoder_output.mean(dim=1)  # (batch_size, hidden_dim)

        # Project to semantic space with adapter
        semantic_vector = self.adapter(pooled_output)

        return semantic_vector

# Encoder Pipeline, with the adapter as the first, or last step of the pipeline.
# class TextPipeline(Pipeline):
#     def _sanitize_parameters(self, **kwargs, adapter: str):
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

