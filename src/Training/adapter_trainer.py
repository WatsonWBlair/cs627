"""
Lightweight trainer for adapter-only training with pre-generated tokens.

This module provides a simplified trainer that works exclusively with
pre-computed encoder tokens, enabling fast adapter optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import numpy as np


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
            
            # Mask out self-similarity
            mask = torch.eye(batch_size, device=query.device, dtype=torch.bool)
            neg_sims.masked_fill_(mask, -1e9)
        else:
            negatives = F.normalize(negatives, p=2, dim=1)
            neg_sims = torch.mm(query, negatives.t()) / self.temperature  # (B, N)
        
        # Compute InfoNCE loss
        # For each query, maximize similarity to positives, minimize to negatives
        pos_exp = torch.exp(pos_sims).mean(dim=1)  # Average over positives
        neg_exp = torch.exp(neg_sims).sum(dim=1)  # Sum over negatives
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
        
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
    """
    
    def __init__(
        self,
        adapters: Dict[str, nn.Module],
        loss_fn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device = torch.device('cpu'),
        use_amp: bool = False,
        gradient_accumulation: int = 1
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
        """
        self.adapters = adapters
        self.loss_fn = loss_fn
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation = gradient_accumulation
        
        # Move modules to device
        for adapter in self.adapters.values():
            adapter.to(device)
        self.loss_fn.to(device)
        
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
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        # Set to train mode
        for adapter in self.adapters.values():
            adapter.train()
        
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
                    # Find corresponding tokens
                    encoder_name = name.replace('_adapter', '').replace('adapter', '')
                    if encoder_name in tokens:
                        with autocast(enabled=self.use_amp):
                            embeddings[name] = adapter(tokens[encoder_name])
                
                # Compute contrastive loss between all pairs
                loss = 0
                num_pairs = 0
                for name1, emb1 in embeddings.items():
                    positives = [emb2 for name2, emb2 in embeddings.items() if name2 != name1]
                    if positives:
                        with autocast(enabled=self.use_amp):
                            pair_loss = self.loss_fn(emb1, positives)
                        loss += pair_loss
                        num_pairs += 1
                
                if num_pairs > 0:
                    loss = loss / num_pairs
                
            else:
                # Reconstruction learning
                with autocast(enabled=self.use_amp):
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
            
            # Track loss
            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation:.4f}'})
        
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
                        encoder_name = name.replace('_adapter', '').replace('adapter', '')
                        if encoder_name in tokens:
                            emb = adapter(tokens[encoder_name])
                            embeddings[name] = emb
                            all_embeddings[name].append(emb.cpu())
                    
                    # Compute loss
                    loss = 0
                    num_pairs = 0
                    for name1, emb1 in embeddings.items():
                        positives = [emb2 for name2, emb2 in embeddings.items() if name2 != name1]
                        if positives:
                            pair_loss = self.loss_fn(emb1, positives)
                            loss += pair_loss
                            num_pairs += 1
                    
                    if num_pairs > 0:
                        loss = loss / num_pairs
                    
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