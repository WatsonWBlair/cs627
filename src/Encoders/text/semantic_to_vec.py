import torch
from transformers import BartModel, BartTokenizer
from typing import Union, Optional
from src.utils.Adapter import Adapter

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

        # Load trained adapter weights if they exist
        try:
            self.adapter.load(device=DEVICE)
            print(f"[Text_to_Vec] [OK] Loaded adapter weights: {self.adapter.weights_path}")
        except FileNotFoundError:
            print(f"[Text_to_Vec] [WARNING] No trained adapter weights found!")
            print(f"[Text_to_Vec] [WARNING] Expected: {self.adapter.weights_path}")
            print(f"[Text_to_Vec] [WARNING] Encoder will NOT work until trained.")
            print(f"[Text_to_Vec] [WARNING] Train using: python src/Training/train_text_encoder.py")

    def forward(
        self,
        input_text: Union[str, list[str]],
        *,
        max_length: Optional[int] = None,
        device: str = DEVICE,
        pregen: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: tokenize -> encode with BART -> project with adapter

        Args:
            input_text: String or list of strings
            max_length: Maximum sequence length
            device: Device to use
            pregen: Pre-Generation mode, used to pre-generate training data. 
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

        # If in pregeneration mode
        if(pregen):
            return pooled_output

        # Project to semantic space with adapter
        semantic_vector = self.adapter(pooled_output)

        return semantic_vector
