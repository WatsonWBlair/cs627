import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path


class Adapter(nn.Module):
    """
    MLP Adapter for translating between modality spaces and semantic vector space.

    Args:
        prefix: Unique identifier for saving/loading weights
        input_length: Input dimension (default: 1024)
        output_length: Output dimension (default: 1024)
        hidden_size: Hidden layer dimension (default: 200)
        hidden_layers: Number of hidden layers (default: 2)
        weights_dir: Directory to save/load weights (default: "OptimalWeights")
    """

    config_class: str = 'NLP'

    def __init__(
        self,
        prefix: str,
        input_length: int = 1024,
        output_length: int = 1024,
        hidden_size: int = 200,
        hidden_layers: int = 2,
        weights_dir: str = "OptimalWeights"
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.weights_path = Path(f"{weights_dir}/{prefix}_weights.pth")

        # Build MLP layers
        model_layers: list[nn.Module] = [
            nn.Linear(input_length, hidden_size),
            nn.ReLU()
        ]

        for _ in range(hidden_layers):
            model_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

        model_layers.append(nn.Linear(hidden_size, output_length))

        self.model = nn.Sequential(*model_layers)

    def load(self, device: Optional[str] = None) -> None:
        """
        Load adapter weights from disk.

        Args:
            device: Device to load weights to (default: None, uses CPU)
        """
        weights = torch.load(
            self.weights_path,
            map_location=device or 'cpu'
        )
        self.model.load_state_dict(weights)

    def save(self) -> None:
        """Save adapter weights to disk."""
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)

    def forward(
        self,
        input_vectors: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through adapter.

        Args:
            input_vectors: Input tensor of shape (batch_size, input_length)
            labels: Optional labels (unused, kept for compatibility)

        Returns:
            Output tensor of shape (batch_size, output_length)
        """
        return self.model(input_vectors)