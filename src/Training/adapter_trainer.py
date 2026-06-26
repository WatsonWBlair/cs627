"""
Lightweight trainer for adapter-only training with pre-generated tokens.

This module provides a simplified trainer that works exclusively with
pre-computed encoder tokens, enabling fast adapter optimization.

Supports MoCo (Momentum Contrast) for improved contrastive learning:
- Momentum adapters: Slow-moving copies for stable representations
- Negative queue: Large pool of negatives from past batches
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import numpy as np


class MomentumQueue:
    """
    Queue for storing negative sample embeddings from past batches.

    Used in MoCo-style contrastive learning to provide a large pool of
    negatives without requiring large batch sizes.

    Args:
        queue_size: Maximum number of embeddings to store
        dim: Embedding dimension (default: 1024)
        device: Device to store queue on
    """

    def __init__(self, queue_size: int, dim: int = 1024, device: str = 'cpu'):
        self.queue_size = queue_size
        self.dim = dim
        self.device = device

        # Initialize with random normalized vectors
        self.queue = F.normalize(
            torch.randn(queue_size, dim), dim=1
        ).to(device)
        self.ptr = 0
        self._is_full = False

    @torch.no_grad()
    def update(self, vectors: torch.Tensor) -> None:
        """
        Add new vectors to the queue, replacing oldest entries.

        Args:
            vectors: New embeddings to add (B, D)
        """
        vectors = F.normalize(vectors.detach(), dim=1)
        batch_size = vectors.shape[0]

        if batch_size >= self.queue_size:
            # Replace entire queue with last queue_size vectors
            self.queue = vectors[-self.queue_size:].clone()
            self.ptr = 0
            self._is_full = True
        elif self.ptr + batch_size <= self.queue_size:
            # Simple case: fits without wrapping
            self.queue[self.ptr:self.ptr + batch_size] = vectors
            self.ptr = self.ptr + batch_size
            if self.ptr == self.queue_size:
                self._is_full = True
                self.ptr = 0
        else:
            # Wrap around
            remaining = self.queue_size - self.ptr
            self.queue[self.ptr:] = vectors[:remaining]
            self.queue[:batch_size - remaining] = vectors[remaining:]
            self.ptr = batch_size - remaining
            self._is_full = True

    def get_queue(self) -> torch.Tensor:
        """
        Get current queue contents.

        Returns:
            Queue embeddings (queue_size, D) or (ptr, D) if not full
        """
        if self._is_full:
            return self.queue.clone()
        else:
            # Return only filled portion
            return self.queue[:self.ptr].clone() if self.ptr > 0 else None

    def is_ready(self, min_samples: int = 256) -> bool:
        """Check if queue has enough samples for meaningful negatives."""
        return self._is_full or self.ptr >= min_samples


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for cross-modal alignment.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        positives: List[torch.Tensor],
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            query: Query embeddings (B, D)
            positives: List of positive embeddings
            negatives: Negative embeddings (N, D) or None for in-batch negatives
        
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        positives = [F.normalize(p, p=2, dim=1) for p in positives]
        
        # Compute positive similarities
        pos_sims = []
        for pos in positives:
            sim = torch.sum(query * pos, dim=1) / self.temperature
            pos_sims.append(sim)
        pos_sims = torch.stack(pos_sims, dim=1)  # (B, num_positives)
        
        # Use in-batch negatives if not provided
        if negatives is None:
            # All other samples in batch are negatives
            batch_size = query.shape[0]
            neg_sims = torch.mm(query, query.t()) / self.temperature  # (B, B)
            
            # Mask out self-similarity (use -inf for proper softmax behavior)
            mask = torch.eye(batch_size, device=query.device, dtype=torch.bool)
            neg_sims.masked_fill_(mask, float('-inf'))
        else:
            negatives = F.normalize(negatives, p=2, dim=1)
            neg_sims = torch.mm(query, negatives.t()) / self.temperature  # (B, N)
        
        # Compute InfoNCE loss using numerically stable log-sum-exp
        # This avoids float16 overflow when queue_size is large
        pos_mean = pos_sims.mean(dim=1, keepdim=True)  # (B, 1)
        logits = torch.cat([pos_mean, neg_sims], dim=1)  # (B, 1 + N)

        # Labels: positive is at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class ReconstructionLoss(nn.Module):
    """
    MSE reconstruction loss for decoder training.
    """
    
    def __init__(self, cosine_weight: float = 0.1):
        super().__init__()
        self.cosine_weight = cosine_weight
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss with optional cosine similarity term.
        
        Args:
            predicted: Predicted embeddings (B, D)
            target: Target embeddings (B, D)
        
        Returns:
            Reconstruction loss value
        """
        # MSE loss
        mse_loss = F.mse_loss(predicted, target)
        
        # Cosine similarity loss (encourage directional alignment)
        if self.cosine_weight > 0:
            pred_norm = F.normalize(predicted, p=2, dim=1)
            target_norm = F.normalize(target, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
            cosine_loss = 1 - cosine_sim  # Convert similarity to loss
            
            total_loss = mse_loss + self.cosine_weight * cosine_loss
        else:
            total_loss = mse_loss
        
        return total_loss


class AdapterTrainer:
    """
    Trainer for adapter-only training with pre-generated tokens.

    Supports MoCo (Momentum Contrast) for improved contrastive learning:
    - Momentum adapters: Slow-moving copies updated via EMA
    - Negative queue: Large pool of negatives from past batches
    """

    def __init__(
        self,
        adapters: Dict[str, nn.Module],
        loss_fn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device = torch.device('cpu'),
        use_amp: bool = False,
        gradient_accumulation: int = 1,
        encoder_mapping: Optional[Dict[str, str]] = None,
        # MoCo parameters
        use_moco: bool = True,
        queue_size: int = 4096,
        momentum: float = 0.999,
        embedding_dim: int = 1024
    ):
        """
        Initialize adapter trainer.

        Args:
            adapters: Dictionary of adapter modules
            loss_fn: Loss function (ContrastiveLoss or ReconstructionLoss)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            device: Device for training
            use_amp: Use automatic mixed precision
            gradient_accumulation: Gradient accumulation steps
            encoder_mapping: Maps adapter name to encoder/token name
            use_moco: Enable MoCo (momentum adapters + queue)
            queue_size: Size of negative sample queue
            momentum: Momentum coefficient for EMA update (0.999 = slow update)
            embedding_dim: Dimension of embeddings (for queue)
        """
        self.adapters = adapters
        self.loss_fn = loss_fn
        self.device = device
        # AMP only works with CUDA - disable on CPU
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation = gradient_accumulation
        self.encoder_mapping = encoder_mapping or {}

        # MoCo settings
        self.use_moco = use_moco and isinstance(loss_fn, ContrastiveLoss)
        self.momentum = momentum

        # Move modules to device
        for adapter in self.adapters.values():
            adapter.to(device)
        self.loss_fn.to(device)

        # Create momentum adapters (MoCo)
        if self.use_moco:
            self.momentum_adapters = {}
            for name, adapter in self.adapters.items():
                # Deep copy the adapter
                mom_adapter = copy.deepcopy(adapter)
                mom_adapter.to(device)
                # Freeze momentum adapter (no gradients)
                for param in mom_adapter.parameters():
                    param.requires_grad = False
                self.momentum_adapters[name] = mom_adapter

            # Create negative queue
            self.queue = MomentumQueue(
                queue_size=queue_size,
                dim=embedding_dim,
                device=str(device)
            )
            print(f"[MoCo] Enabled with queue_size={queue_size}, momentum={momentum}")
        else:
            self.momentum_adapters = None
            self.queue = None
            if isinstance(loss_fn, ContrastiveLoss):
                print("[MoCo] Disabled - using in-batch negatives only")

        # Create optimizer for all adapter parameters
        all_params = []
        for adapter in self.adapters.values():
            all_params.extend(adapter.parameters())

        self.optimizer = optim.AdamW(
            all_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Mixed precision scaler (only when CUDA is available)
        self.scaler = GradScaler('cuda') if self.use_amp else None

    @torch.no_grad()
    def _momentum_update(self) -> None:
        """
        Update momentum adapters using exponential moving average.

        Formula: θ_m = m * θ_m + (1 - m) * θ
        Where m is momentum (e.g., 0.999 means very slow updates)
        """
        if not self.use_moco:
            return

        for name in self.adapters:
            adapter = self.adapters[name]
            mom_adapter = self.momentum_adapters[name]

            for param, mom_param in zip(
                adapter.parameters(),
                mom_adapter.parameters()
            ):
                mom_param.data = (
                    self.momentum * mom_param.data +
                    (1.0 - self.momentum) * param.data
                )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.

        With MoCo enabled:
        1. Forward pass through adapters (query embeddings)
        2. Forward pass through momentum adapters (key embeddings, no grad)
        3. Compute loss with queue negatives
        4. Update queue with momentum embeddings
        5. Update momentum adapters after optimizer step

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        # Set to train mode
        for adapter in self.adapters.values():
            adapter.train()

        # Momentum adapters stay in eval mode
        if self.use_moco:
            for mom_adapter in self.momentum_adapters.values():
                mom_adapter.eval()

        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if 'tokens' in batch:
                # Multi-encoder tokens
                tokens = {
                    name: t.to(self.device)
                    for name, t in batch['tokens'].items()
                }
            else:
                # Single source tokens
                tokens = batch['source_tokens'].to(self.device)

            # Forward pass through adapters
            if isinstance(self.loss_fn, ContrastiveLoss):
                # Contrastive learning: process each encoder's tokens
                embeddings = {}
                for name, adapter in self.adapters.items():
                    # Find corresponding tokens using encoder_mapping
                    encoder_name = self.encoder_mapping.get(name, name.replace('_adapter', ''))
                    if encoder_name in tokens:
                        with autocast('cuda', enabled=self.use_amp):
                            embeddings[name] = adapter(tokens[encoder_name])

                # MoCo: Get momentum embeddings and queue negatives
                if self.use_moco:
                    # Forward pass through momentum adapters (no gradients)
                    with torch.no_grad():
                        mom_embeddings = {}
                        for name, mom_adapter in self.momentum_adapters.items():
                            encoder_name = self.encoder_mapping.get(name, name.replace('_adapter', ''))
                            if encoder_name in tokens:
                                mom_embeddings[name] = mom_adapter(tokens[encoder_name])

                    # Get negatives from queue (if ready)
                    queue_negatives = self.queue.get_queue() if self.queue.is_ready() else None
                else:
                    mom_embeddings = None
                    queue_negatives = None

                # Compute contrastive loss between all pairs
                losses = []
                for name1, emb1 in embeddings.items():
                    positives = [emb2 for name2, emb2 in embeddings.items() if name2 != name1]
                    if positives:
                        with autocast('cuda', enabled=self.use_amp):
                            # Pass queue negatives to loss function
                            pair_loss = self.loss_fn(emb1, positives, queue_negatives)
                        losses.append(pair_loss)

                if losses:
                    loss = torch.stack(losses).mean()
                else:
                    # No valid pairs - skip this batch
                    continue

                # MoCo: Update queue with momentum embeddings
                if self.use_moco and mom_embeddings:
                    # Concatenate all momentum embeddings and add to queue
                    all_mom_embs = torch.cat(list(mom_embeddings.values()), dim=0)
                    self.queue.update(all_mom_embs)

            else:
                # Reconstruction learning (MoCo not applicable)
                with autocast('cuda', enabled=self.use_amp):
                    # For simplicity, use first adapter
                    adapter = list(self.adapters.values())[0]
                    predicted = adapter(tokens)

                    # Target is the same tokens (autoencoder style)
                    loss = self.loss_fn(predicted, tokens)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # MoCo: Update momentum adapters after optimizer step
                self._momentum_update()

            # Track loss
            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1

            # Update progress bar
            postfix = {'loss': f'{loss.item() * self.gradient_accumulation:.4f}'}
            if self.use_moco:
                postfix['queue'] = f'{self.queue.ptr}/{self.queue.queue_size}'
            pbar.set_postfix(postfix)

        # Step scheduler
        self.scheduler.step()
        
        return total_loss / num_batches
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate on a dataset.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dictionary of validation metrics
        """
        # Set to eval mode
        for adapter in self.adapters.values():
            adapter.eval()
        
        total_loss = 0
        num_batches = 0
        
        # For computing additional metrics
        all_embeddings = {name: [] for name in self.adapters.keys()}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                if 'tokens' in batch:
                    tokens = {
                        name: t.to(self.device)
                        for name, t in batch['tokens'].items()
                    }
                else:
                    tokens = batch['source_tokens'].to(self.device)
                
                # Forward pass
                if isinstance(self.loss_fn, ContrastiveLoss):
                    embeddings = {}
                    for name, adapter in self.adapters.items():
                        encoder_name = self.encoder_mapping.get(name, name.replace('_adapter', ''))
                        if encoder_name in tokens:
                            emb = adapter(tokens[encoder_name])
                            embeddings[name] = emb
                            all_embeddings[name].append(emb.cpu())

                    # Compute loss
                    losses = []
                    for name1, emb1 in embeddings.items():
                        positives = [emb2 for name2, emb2 in embeddings.items() if name2 != name1]
                        if positives:
                            pair_loss = self.loss_fn(emb1, positives)
                            losses.append(pair_loss)

                    if losses:
                        loss = torch.stack(losses).mean()
                    else:
                        continue
                    
                else:
                    adapter = list(self.adapters.values())[0]
                    predicted = adapter(tokens)
                    loss = self.loss_fn(predicted, tokens)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        metrics = {
            'loss': total_loss / num_batches
        }
        
        # Compute recall@K for contrastive learning
        if isinstance(self.loss_fn, ContrastiveLoss) and len(all_embeddings) > 1:
            # Concatenate all embeddings
            for name in all_embeddings:
                if all_embeddings[name]:
                    all_embeddings[name] = torch.cat(all_embeddings[name], dim=0)
            
            # Compute recall between first two modalities
            if len(all_embeddings) >= 2:
                names = list(all_embeddings.keys())[:2]
                emb1 = F.normalize(all_embeddings[names[0]], p=2, dim=1)
                emb2 = F.normalize(all_embeddings[names[1]], p=2, dim=1)
                
                # Compute similarities
                similarities = torch.mm(emb1, emb2.t())
                
                # Compute recall@K
                _, indices = torch.sort(similarities, dim=1, descending=True)
                ground_truth = torch.arange(emb1.shape[0])
                
                for k in [1, 5]:
                    correct = (indices[:, :k] == ground_truth.unsqueeze(1)).any(dim=1)
                    recall = correct.float().mean().item()
                    metrics[f'recall@{k}'] = recall
        
        # Compute cosine similarity for reconstruction
        elif isinstance(self.loss_fn, ReconstructionLoss):
            # Would need target embeddings to compute this properly
            # For now, just return loss
            pass
        
        return metrics
    
    def save_adapters(self):
        """Save all adapter weights."""
        for name, adapter in self.adapters.items():
            if hasattr(adapter, 'save'):
                adapter.save()
            else:
                # Fallback to manual save
                torch.save(
                    adapter.state_dict(),
                    f"OptimalWeights/{name}_weights.pth"
                )
    
    def load_adapters(self):
        """Load all adapter weights."""
        for name, adapter in self.adapters.items():
            if hasattr(adapter, 'load'):
                adapter.load(self.device)
            else:
                # Fallback to manual load
                adapter.load_state_dict(
                    torch.load(
                        f"OptimalWeights/{name}_weights.pth",
                        map_location=self.device
                    )
                )