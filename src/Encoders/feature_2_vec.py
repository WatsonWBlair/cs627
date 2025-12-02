"""
Simple feature-to-vector encoders for pre-extracted features.

These are lightweight encoders that work with pre-extracted features (like COVAREP audio
or FACET video features) and map them to the semantic vector space using only an MLP adapter.

This is useful when raw modality data (audio waveforms, video frames) is not available.
"""

import torch
from utils.Adapter import Adapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Feature_to_Vec(torch.nn.Module):
    """
    Generic encoder for pre-extracted features

    Maps pre-extracted numerical features to semantic vector space using an MLP adapter.

    Args:
        feature_name: Name for the feature type (used in adapter prefix)
        input_dim: Dimension of input features
        output_dim: Dimension of output semantic vectors (default: 1024)
        hidden_size: Size of adapter hidden layers (default: 200)
        hidden_layers: Number of adapter hidden layers (default: 2)
    """

    def __init__(
        self,
        feature_name: str,
        input_dim: int,
        output_dim: int = 1024,
        hidden_size: int = 200,
        hidden_layers: int = 2
    ):
        super(Feature_to_Vec, self).__init__()

        self.feature_name = feature_name
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create adapter to map features to semantic space
        self.adapter = Adapter(
            prefix=f"{feature_name}_feature_enc",
            input_length=input_dim,
            output_length=output_dim,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers
        )

    def forward(self, features, device: str = DEVICE):
        """
        Forward pass: features -> adapter -> semantic vector

        Args:
            features: Input features - either:
                     - Tensor (batch_size, input_dim) or (batch_size, seq_len, input_dim)
                     - List of tensors [(seq_len_1, input_dim), (seq_len_2, input_dim), ...]
            device: Device to use

        Returns:
            Semantic vector representation (batch_size, output_dim)
        """
        # Handle list of variable-length tensors (from custom collate_fn)
        if isinstance(features, list):
            # Process each sequence, apply mean pooling, then stack
            pooled_features = []
            for feat in features:
                if not isinstance(feat, torch.Tensor):
                    feat = torch.tensor(feat, dtype=torch.float32)
                feat = feat.to(device)

                # Mean pool over sequence dimension if needed
                if len(feat.shape) == 2:  # (seq_len, feature_dim)
                    feat = feat.mean(dim=0)  # (feature_dim,)

                pooled_features.append(feat)

            # Stack into batch
            features = torch.stack(pooled_features, dim=0)  # (batch_size, feature_dim)
        else:
            # Handle regular tensor input
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            features = features.to(device)

            # If features have sequence dimension, average pool them
            if len(features.shape) == 3:  # (batch, seq_len, features)
                features = features.mean(dim=1)  # (batch, features)

        # Project to semantic space with adapter
        semantic_vector = self.adapter(features)

        return semantic_vector


# Pre-configured encoders for common feature types

class Text_to_Vec_Features(Feature_to_Vec):
    """
    Text feature encoder (for GloVe features)

    GloVe features are 300-dimensional word embeddings.
    """
    def __init__(self, output_dim: int = 1024):
        super().__init__(
            feature_name="glove_text",
            input_dim=300,  # GloVe feature dimension
            output_dim=output_dim
        )


class Audio_to_Vec_Features(Feature_to_Vec):
    """
    Audio feature encoder (for COVAREP features)

    COVAREP features are 74-dimensional acoustic features.
    """
    def __init__(self, output_dim: int = 1024):
        super().__init__(
            feature_name="covarep_audio",
            input_dim=74,  # COVAREP feature dimension
            output_dim=output_dim
        )


class Video_to_Vec_Features(Feature_to_Vec):
    """
    Video feature encoder (for FACET features)

    FACET 4.2 features are 35-dimensional facial expression features.
    """
    def __init__(self, output_dim: int = 1024):
        super().__init__(
            feature_name="facet_video",
            input_dim=35,  # FACET 4.2 feature dimension
            output_dim=output_dim
        )
