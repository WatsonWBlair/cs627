"""
Experiment Runner for individual factorization study experiments.

Handles:
- Creating configurable encoders with custom adapter configurations
- Running training with momentum contrastive learning
- Collecting metrics during training
- Saving results and checkpoints
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from pathlib import Path
from tqdm import tqdm
import copy
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.FactorizationStudy.config.study_config import (
    ExperimentConfig,
    StudyConfig,
    AdapterConfig,
    InitStrategy,
)
from src.FactorizationStudy.adapters.configurable_encoders import (
    ConfigurableTextEncoder,
    ConfigurableAudioEncoder,
    ConfigurableImageEncoder,
)
from src.FactorizationStudy.runner.resource_monitor import ResourceMonitor, ResourceSummary

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EpochMetrics:
    """Metrics collected during a single epoch."""
    epoch: int
    total_loss: float
    encoder_losses: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    epoch_time_seconds: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class ExperimentResults:
    """Results from a completed experiment."""
    experiment_id: str
    config: Dict[str, Any]

    # Training history
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)

    # Final metrics
    final_loss: float = 0.0
    best_loss: float = float('inf')
    best_epoch: int = 0

    # Evaluation metrics
    semantic_fidelity: Dict[str, float] = field(default_factory=dict)
    cross_modal_retrieval: Dict[str, float] = field(default_factory=dict)

    # Resource usage
    resource_summary: Optional[ResourceSummary] = None

    # Status
    completed: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config,
            "epoch_metrics": [
                {
                    "epoch": m.epoch,
                    "total_loss": m.total_loss,
                    "encoder_losses": m.encoder_losses,
                    "learning_rate": m.learning_rate,
                    "epoch_time_seconds": m.epoch_time_seconds,
                    "gpu_memory_mb": m.gpu_memory_mb,
                }
                for m in self.epoch_metrics
            ],
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "semantic_fidelity": self.semantic_fidelity,
            "cross_modal_retrieval": self.cross_modal_retrieval,
            "resource_summary": self.resource_summary.to_dict() if self.resource_summary else None,
            "completed": self.completed,
            "error_message": self.error_message,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MultiPositiveInfoNCE(nn.Module):
    """InfoNCE loss with multiple positives per query."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        positives: List[torch.Tensor],
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with multiple positives.

        Args:
            query: Query vectors (batch_size, dim)
            positives: List of positive vectors, each (batch_size, dim)
            negatives: Negative vectors from queue (queue_size, dim)

        Returns:
            Loss scalar
        """
        query = F.normalize(query, dim=1)
        positives = [F.normalize(pos, dim=1) for pos in positives]
        negatives = F.normalize(negatives, dim=1)

        # Compute similarity with positives
        pos_sims = []
        for pos in positives:
            sim = torch.matmul(query, pos.t()) / self.temperature
            pos_sims.append(torch.diagonal(sim))

        pos_sims = torch.stack(pos_sims, dim=1)

        # Compute similarity with negatives
        neg_sims = torch.matmul(query, negatives.t()) / self.temperature

        # Loss: -log(sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))))
        numerator = torch.sum(torch.exp(pos_sims), dim=1)
        denominator = numerator + torch.sum(torch.exp(neg_sims), dim=1)
        loss = -torch.log(numerator / denominator)

        return loss.mean()


class MomentumQueue:
    """Queue for storing negative samples."""

    def __init__(self, queue_size: int, dim: int, device: str):
        self.queue_size = queue_size
        self.dim = dim
        self.device = device
        self.queue = F.normalize(torch.randn(queue_size, dim), dim=1).to(device)
        self.ptr = 0

    @torch.no_grad()
    def update(self, vectors: torch.Tensor):
        """Update queue with new vectors."""
        batch_size = vectors.shape[0]
        if self.ptr + batch_size <= self.queue_size:
            self.queue[self.ptr:self.ptr + batch_size] = vectors
            self.ptr = (self.ptr + batch_size) % self.queue_size
        else:
            remaining = self.queue_size - self.ptr
            self.queue[self.ptr:] = vectors[:remaining]
            self.queue[:batch_size - remaining] = vectors[remaining:]
            self.ptr = batch_size - remaining

    def get_queue(self) -> torch.Tensor:
        return self.queue.clone()


class ExperimentRunner:
    """
    Runs a single factorization study experiment.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        study_config: StudyConfig,
        data_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = DEVICE,
    ):
        """
        Initialize experiment runner.

        Args:
            experiment_config: Configuration for this experiment
            study_config: Overall study configuration
            data_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to run on
        """
        self.exp_config = experiment_config
        self.study_config = study_config
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.device = device

        # Training hyperparameters
        self.epochs = experiment_config.epochs or study_config.default_epochs
        self.batch_size = experiment_config.batch_size or study_config.default_batch_size
        self.learning_rate = experiment_config.learning_rate or study_config.default_learning_rate

        # Create output directory
        self.output_dir = study_config.get_experiment_dir(experiment_config.experiment_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.encoders = {}
        self.encoders_m = {}  # Momentum encoders
        self.queues = {}
        self.resource_monitor = ResourceMonitor(device)

        # Results
        self.results = ExperimentResults(
            experiment_id=experiment_config.experiment_id,
            config=experiment_config.to_dict(),
        )

    def setup_encoders(self) -> None:
        """Create and initialize configurable encoders."""
        semantic_dim = self.exp_config.semantic_dim
        adapter_config = self.exp_config.encoder_adapter
        init_strategy = self.exp_config.init_strategy
        weights_dir = str(self.output_dir / "weights")

        print(f"\n[ExperimentRunner] Setting up encoders:")
        print(f"  Semantic dim: {semantic_dim}")
        print(f"  Adapter: h{adapter_config.hidden_size}_l{adapter_config.hidden_layers}")
        print(f"  Init strategy: {init_strategy.value}")

        encoder_configs = []

        # Text encoder
        if self.exp_config.train_text:
            text_encoder = ConfigurableTextEncoder(
                semantic_dim=semantic_dim,
                adapter_config=adapter_config,
                weights_dir=weights_dir,
                experiment_id=self.exp_config.experiment_id,
                init_strategy=init_strategy,
                load_weights=False,  # Start fresh for study
                verbose=True,
            )
            self.encoders["text"] = text_encoder.to(self.device)
            encoder_configs.append(("text", "text"))

        # Audio encoder
        if self.exp_config.train_audio:
            audio_encoder = ConfigurableAudioEncoder(
                semantic_dim=semantic_dim,
                adapter_config=adapter_config,
                weights_dir=weights_dir,
                experiment_id=self.exp_config.experiment_id,
                init_strategy=init_strategy,
                load_weights=False,
                verbose=True,
            )
            self.encoders["audio"] = audio_encoder.to(self.device)
            encoder_configs.append(("audio", "audio"))

        # Image encoder
        if self.exp_config.train_image:
            image_encoder = ConfigurableImageEncoder(
                semantic_dim=semantic_dim,
                adapter_config=adapter_config,
                weights_dir=weights_dir,
                experiment_id=self.exp_config.experiment_id,
                init_strategy=init_strategy,
                load_weights=False,
                verbose=True,
            )
            self.encoders["image"] = image_encoder.to(self.device)
            encoder_configs.append(("image", "video"))

        # Create momentum encoders and queues
        for name, _ in encoder_configs:
            encoder = self.encoders[name]

            # Momentum encoder
            encoder_m = copy.deepcopy(encoder).to(self.device)
            for param in encoder_m.parameters():
                param.requires_grad = False
            self.encoders_m[name] = encoder_m

            # Queue
            self.queues[name] = MomentumQueue(4096, semantic_dim, self.device)

        # Create optimizer for adapters
        adapter_params = []
        for name, encoder in self.encoders.items():
            if hasattr(encoder, 'adapter'):
                adapter_params.extend(encoder.adapter.parameters())

        self.optimizer = torch.optim.Adam(adapter_params, lr=self.learning_rate)
        self.criterion = MultiPositiveInfoNCE(temperature=0.07)

        print(f"[ExperimentRunner] Created {len(self.encoders)} encoders")

    @torch.no_grad()
    def update_momentum_encoders(self, momentum: float = 0.999):
        """Update momentum encoders with EMA."""
        for name in self.encoders:
            encoder_q = self.encoders[name]
            encoder_k = self.encoders_m[name]
            for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of losses
        """
        for encoder in self.encoders.values():
            encoder.train()

        total_loss = 0.0
        encoder_losses = {name: 0.0 for name in self.encoders}
        num_batches = 0

        for batch in tqdm(self.data_loader, desc=f"Epoch {epoch}", leave=False):
            # Encode with query encoders
            query_outputs = {}
            for name, encoder in self.encoders.items():
                data_key = "video" if name == "image" else name
                if data_key in batch and batch[data_key] is not None:
                    data = batch[data_key]
                    query_outputs[name] = encoder(data, device=self.device)

            if len(query_outputs) < 2:
                continue

            # Encode with momentum encoders
            with torch.no_grad():
                key_outputs = {}
                for name, encoder in self.encoders_m.items():
                    data_key = "video" if name == "image" else name
                    if data_key in batch and batch[data_key] is not None:
                        data = batch[data_key]
                        key_outputs[name] = encoder(data, device=self.device)

            # Compute loss for each encoder
            batch_losses = {}
            for name in query_outputs:
                query_vec = query_outputs[name]

                # Positives: other encoders from same segment
                positive_vecs = [
                    key_outputs[other_name]
                    for other_name in key_outputs
                    if other_name != name
                ]

                if not positive_vecs:
                    continue

                # Negatives from queue
                negative_vecs = self.queues[name].get_queue()

                # Compute loss
                loss = self.criterion(query_vec, positive_vecs, negative_vecs)
                batch_losses[name] = loss
                encoder_losses[name] += loss.item()

            if not batch_losses:
                continue

            # Total loss
            batch_total_loss = sum(batch_losses.values())
            total_loss += batch_total_loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            batch_total_loss.backward()
            self.optimizer.step()

            # Update momentum encoders and queues
            self.update_momentum_encoders()
            for name in key_outputs:
                self.queues[name].update(key_outputs[name].detach())

            num_batches += 1

        # Average losses
        if num_batches > 0:
            total_loss /= num_batches
            encoder_losses = {k: v / num_batches for k, v in encoder_losses.items()}

        return {"total_loss": total_loss, **encoder_losses}

    def run(self) -> ExperimentResults:
        """
        Run the complete experiment.

        Returns:
            ExperimentResults with all collected metrics
        """
        try:
            print(f"\n{'='*60}")
            print(f"Starting Experiment: {self.exp_config.experiment_id}")
            print(f"{'='*60}")

            # Setup
            self.setup_encoders()

            # Start monitoring
            self.resource_monitor.start_training()

            best_loss = float('inf')
            best_epoch = 0

            # Training loop
            for epoch in range(1, self.epochs + 1):
                self.resource_monitor.start_epoch(epoch)

                # Train
                losses = self.train_epoch(epoch)

                # Record epoch
                epoch_time = self.resource_monitor.end_epoch()

                epoch_metrics = EpochMetrics(
                    epoch=epoch,
                    total_loss=losses["total_loss"],
                    encoder_losses={k: v for k, v in losses.items() if k != "total_loss"},
                    learning_rate=self.learning_rate,
                    epoch_time_seconds=epoch_time,
                    gpu_memory_mb=self.resource_monitor.get_current_gpu_memory_mb(),
                )
                self.results.epoch_metrics.append(epoch_metrics)

                # Track best
                if losses["total_loss"] < best_loss:
                    best_loss = losses["total_loss"]
                    best_epoch = epoch
                    self._save_best_checkpoint()

                # Log progress
                print(f"Epoch {epoch}/{self.epochs} | Loss: {losses['total_loss']:.4f} | "
                      f"Time: {epoch_time:.1f}s | GPU: {epoch_metrics.gpu_memory_mb:.0f}MB")

                # Save checkpoint periodically
                if epoch % self.study_config.checkpoint_interval == 0:
                    self._save_checkpoint(epoch)

            # Finalize
            self.resource_monitor.end_training()

            self.results.final_loss = losses["total_loss"]
            self.results.best_loss = best_loss
            self.results.best_epoch = best_epoch
            self.results.resource_summary = self.resource_monitor.get_summary()
            self.results.completed = True

            # Save final results
            self.results.save(str(self.output_dir / "results.json"))

            print(f"\n[ExperimentRunner] Experiment completed!")
            print(f"  Final loss: {self.results.final_loss:.4f}")
            print(f"  Best loss: {best_loss:.4f} (epoch {best_epoch})")

            return self.results

        except Exception as e:
            self.results.error_message = str(e)
            self.results.completed = False
            print(f"\n[ExperimentRunner] ERROR: {e}")
            raise

    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        for name, encoder in self.encoders.items():
            if hasattr(encoder, 'adapter'):
                path = checkpoint_dir / f"{name}_adapter_epoch{epoch}.pth"
                torch.save(encoder.adapter.model.state_dict(), path)

    def _save_best_checkpoint(self) -> None:
        """Save best model checkpoint."""
        best_dir = self.output_dir / "best"
        best_dir.mkdir(exist_ok=True)

        for name, encoder in self.encoders.items():
            if hasattr(encoder, 'adapter'):
                path = best_dir / f"{name}_adapter_best.pth"
                torch.save(encoder.adapter.model.state_dict(), path)
