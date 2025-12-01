import torch
from transformers import BartModel, BartTokenizer

from utils.Adapter import Adapter

BASE_MODEL = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained Encoder + MLP Adapter architecture for semantic space alignment
class Text_to_Vec(torch.nn.Module):
    def __init__(self, base_model: str = BASE_MODEL, token_length: int = 1024) -> None:
        super(Text_to_Vec, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(base_model)
        self.max_length = token_length

        # Load pretrained BART encoder
        self.encoder = BartModel.from_pretrained(base_model)

        # Create adapter with proper parameters
        self.adapter = Adapter(
            prefix=f"{base_model.replace('/', '_')}_text_enc",
            input_length=token_length,
            output_length=token_length,
            hidden_size=200,
            hidden_layers=2
        )

    def forward(self, input_text, max_length: int = None, device: str = DEVICE):
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

