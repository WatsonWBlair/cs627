import torch
from torch import nn
from transformers import Trainer
from torch.nn import CosineEmbeddingLoss
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import copy


"""
This file contains custom trainers for aligning Encoders to the shared semantic space.

Implements Cross-Modal Momentum Contrastive Learning (MoCo) based on:
- He et al. "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo, CVPR 2020)
- Cross-Modal Alignment for End-to-End Spoken Language Understanding (IEEE 2024)
  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10448143

Key components:
1. Momentum Encoder: Slowly-updated copy of encoder for stable features
2. Memory Queue: Large bank of negative samples across batches
3. InfoNCE Loss: Temperature-scaled contrastive loss
"""


class MultiModalWrapper(nn.Module):
    """
    Wrapper module that contains multiple encoder models in a ModuleDict.

    This allows the HuggingFace Trainer to properly handle a dictionary of models
    by wrapping them in an nn.Module container that supports .to(device), .train(), etc.

    Args:
        encoders: Dictionary of encoder models {'text': encoder, 'audio': encoder, ...}
    """
    def __init__(self, encoders: Dict[str, nn.Module]):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode each modality with its corresponding encoder.

        Args:
            inputs: Dict of inputs {'text': tensor, 'audio': tensor, ...}

        Returns:
            Dict of embeddings {'text': tensor, 'audio': tensor, ...}
        """
        outputs = {}
        for modality, data in inputs.items():
            if modality in self.encoders:
                outputs[modality] = self.encoders[modality](data)
        return outputs


class Contrast(Trainer):
    """
    Cross-Modal Momentum Contrastive Learning Trainer

    Aligns multiple encoders (text, audio, video) to a shared semantic space using
    momentum-based contrastive learning with a memory queue of negative samples.

    Args:
        model: Encoder model or dict of multiple encoders
        momentum: Momentum coefficient for updating momentum encoder (default: 0.999)
        queue_size: Size of negative sample queue (default: 65536)
        temperature: Temperature parameter for InfoNCE loss (default: 0.07)
        embed_dim: Dimension of semantic vectors (default: 1024)
        use_momentum: Whether to use momentum encoder (default: True)
        **kwargs: Additional arguments passed to Trainer

    Example:
        >>> text_encoder = Text_to_Vec()
        >>> audio_encoder = Audio_to_Vec()
        >>> model = {'text': text_encoder, 'audio': audio_encoder}
        >>> trainer = Contrast(
        ...     model=model,
        ...     momentum=0.999,
        ...     queue_size=65536,
        ...     temperature=0.07
        ... )
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        momentum: float = 0.999,
        queue_size: int = 65536,
        temperature: float = 0.07,
        embed_dim: int = 1024,
        use_momentum: bool = True,
        **kwargs
    ):
        # Store original model (dict or module)
        self.is_multimodal = isinstance(model, dict)
        self.original_model = model

        # Wrap dict of models in MultiModalWrapper for HuggingFace Trainer compatibility
        if self.is_multimodal:
            wrapped_model = MultiModalWrapper(model)
        else:
            wrapped_model = model

        super().__init__(model=wrapped_model, **kwargs)

        self.momentum_coef = momentum
        self.temperature = temperature
        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.use_momentum = use_momentum

        # Initialize momentum encoder
        if self.use_momentum:
            if self.is_multimodal:
                # Multiple encoders (multi-modal)
                self.momentum_encoder = {
                    key: copy.deepcopy(encoder) for key, encoder in self.original_model.items()
                }
                # Freeze momentum encoder parameters
                for encoder in self.momentum_encoder.values():
                    for param in encoder.parameters():
                        param.requires_grad = False
            else:
                # Single encoder
                self.momentum_encoder = copy.deepcopy(self.original_model)
                for param in self.momentum_encoder.parameters():
                    param.requires_grad = False

            # Initialize memory queue (stored as regular tensors, not buffers)
            # Queue stores encoded features from momentum encoder (K x D)
            self.queue = F.normalize(torch.randn(queue_size, embed_dim), dim=1)
            self.queue_ptr = torch.zeros(1, dtype=torch.long)

        print(f"Contrast Trainer initialized:")
        print(f"  Momentum: {momentum}")
        print(f"  Queue size: {queue_size}")
        print(f"  Temperature: {temperature}")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Use momentum: {use_momentum}")

    @torch.no_grad()
    def _momentum_update(self):
        """
        Update momentum encoder parameters using exponential moving average

        θ_momentum ← m * θ_momentum + (1 - m) * θ_encoder

        where m is the momentum coefficient (typically 0.999)
        """
        if not self.use_momentum:
            return

        if self.is_multimodal:
            # Update each encoder in the dict
            for key in self.original_model.keys():
                for param_q, param_k in zip(
                    self.original_model[key].parameters(),
                    self.momentum_encoder[key].parameters()
                ):
                    param_k.data = (
                        param_k.data * self.momentum_coef +
                        param_q.data * (1.0 - self.momentum_coef)
                    )
        else:
            # Update single encoder
            for param_q, param_k in zip(
                self.original_model.parameters(),
                self.momentum_encoder.parameters()
            ):
                param_k.data = (
                    param_k.data * self.momentum_coef +
                    param_q.data * (1.0 - self.momentum_coef)
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        Update memory queue with new encoded features (FIFO)

        Handles variable batch sizes (last batch can be smaller) and queue wrapping.

        Args:
            keys: Encoded features from momentum encoder (batch_size x embed_dim)
        """
        if not self.use_momentum:
            return

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Handle wrapping around queue end
        if ptr + batch_size > self.queue_size:
            # Split into two parts if batch wraps around the queue
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
        else:
            # Normal case: batch fits without wrapping
            self.queue[ptr:ptr + batch_size] = keys

        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def compute_infonce_loss(
        self,
        query: torch.Tensor,
        key_positive: torch.Tensor,
        key_negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE (Noise Contrastive Estimation) loss

        Loss = -log(exp(q·k+ / τ) / (exp(q·k+ / τ) + Σ exp(q·k- / τ)))

        where:
            q = query embedding
            k+ = positive key embedding (same sample, different modality)
            k- = negative key embeddings (from queue)
            τ = temperature parameter

        Args:
            query: Query embeddings (batch_size x embed_dim)
            key_positive: Positive key embeddings (batch_size x embed_dim)
            key_negatives: Negative key embeddings (queue_size x embed_dim), optional

        Returns:
            InfoNCE loss (scalar)
        """
        # Normalize embeddings
        query = F.normalize(query, dim=1)
        key_positive = F.normalize(key_positive, dim=1)

        # Compute positive similarity (N x 1)
        pos_sim = torch.einsum('nc,nc->n', [query, key_positive]).unsqueeze(-1)

        if self.use_momentum and key_negatives is not None:
            # Compute negative similarities (N x K)
            neg_sim = torch.einsum('nc,ck->nk', [query, key_negatives.clone().detach().T])

            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim, neg_sim], dim=1)
        else:
            # No queue, only use positive samples
            logits = pos_sim

        # Apply temperature scaling
        logits = logits / self.temperature

        # Labels: positive sample is always first (index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_loss(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        inputs: Union[Dict[str, torch.Tensor], Tuple],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None
    ):
        """
        Compute contrastive loss for cross-modal alignment

        Supports two input formats:
        1. Dict format (for multi-modal learning):
           inputs = {'text': ..., 'audio': ..., 'video': ...}

        2. Tuple format (legacy support for triplet loss):
           inputs = [(query, positive, negative), ...]

        Args:
            model: Encoder model(s)
            inputs: Input data (dict or tuple)
            return_outputs: Whether to return model outputs
            num_items_in_batch: Batch size info (optional)

        Returns:
            loss: Contrastive loss (scalar)
            outputs: Model outputs (if return_outputs=True)
        """

        # Handle legacy tuple format (original triplet loss)
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            if isinstance(inputs[0], tuple):
                return self._compute_triplet_loss(model, inputs, return_outputs)

        # Handle dict format (multi-modal contrastive learning)
        if isinstance(inputs, dict):
            return self._compute_multimodal_loss(model, inputs, return_outputs)

        raise ValueError(
            f"Unsupported input format: {type(inputs)}. "
            "Expected dict or list of tuples."
        )

    def _compute_multimodal_loss(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ):
        """
        Compute cross-modal contrastive loss using InfoNCE

        Aligns multiple modalities (text, audio, video) in shared semantic space

        Args:
            model: Wrapped model or dict of encoders
            inputs: Dict of inputs {'text': tensor, 'audio': tensor, ...}
            return_outputs: Whether to return embeddings

        Returns:
            loss: Combined cross-modal loss
            outputs: Dict of embeddings (if return_outputs=True)
        """
        # Use original_model (the dict of encoders) instead of wrapped model
        encoders = self.original_model if self.is_multimodal else {('default', self.original_model)}
        device = next(iter(encoders.values())).parameters().__next__().device

        # Encode with main model (gradients enabled)
        embeddings = {}
        for modality, encoder in encoders.items():
            if modality in inputs:
                embeddings[modality] = encoder(inputs[modality])

        # Encode with momentum model (no gradients)
        if self.use_momentum:
            with torch.no_grad():
                self._momentum_update()

                momentum_embeddings = {}
                for modality, encoder in self.momentum_encoder.items():
                    if modality in inputs:
                        emb = encoder(inputs[modality])
                        momentum_embeddings[modality] = F.normalize(emb, dim=1)

        # Compute cross-modal losses
        # Example: text ↔ audio, text ↔ video, audio ↔ video
        total_loss = 0.0
        num_pairs = 0

        modalities = list(embeddings.keys())
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i + 1:]:
                # Use mod1 as query, mod2 as key
                query = embeddings[mod1]

                if self.use_momentum:
                    key_positive = momentum_embeddings[mod2]
                    key_negatives = self.queue
                else:
                    key_positive = embeddings[mod2]
                    key_negatives = None

                loss = self.compute_infonce_loss(query, key_positive, key_negatives)
                total_loss += loss
                num_pairs += 1

                # Update queue with current batch
                if self.use_momentum:
                    self._dequeue_and_enqueue(key_positive)

        # Average loss across all modality pairs
        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return (total_loss, embeddings) if return_outputs else total_loss

    def _compute_triplet_loss(
        self,
        model: nn.Module,
        inputs: list[Tuple[str, torch.Tensor, torch.Tensor]],
        return_outputs: bool = False
    ):
        """
        Legacy triplet margin loss (for backward compatibility)

        Loss = max(cos_sim(query, positive) - cos_sim(query, negative) + margin, 0)

        Args:
            model: Encoder model
            inputs: List of (query, positive, negative) tuples
            return_outputs: Whether to return outputs

        Returns:
            loss: Triplet margin loss
            outputs: Model outputs (if return_outputs=True)
        """
        margin = 0.1

        query = inputs[:][:1]  # First item from every tuple
        positive = inputs[:][1:2]  # Second item from every tuple
        negative = inputs[:][-1]  # Third item from every tuple
        outputs = model(*query)

        loss = []
        for q, p, n in zip(outputs, positive, negative):
            maxMe = F.cosine_similarity(q, p)
            minMe = F.cosine_similarity(q, n)
            loss.append(max([maxMe - minMe + margin, 0]))

        return (loss, outputs) if return_outputs else loss


# Add additional training paradigms below:
