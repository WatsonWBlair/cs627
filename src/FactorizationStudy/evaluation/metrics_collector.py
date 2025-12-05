"""
Metrics Collector for aggregating evaluation metrics across experiments.

Collects:
- Training loss and convergence metrics
- Semantic fidelity scores (encode-decode-encode similarity)
- Cross-modal retrieval metrics (Recall@K)
- Resource usage metrics
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path
import json


@dataclass
class EpochMetrics:
    """Metrics collected during a single training epoch."""
    epoch: int
    total_loss: float
    encoder_losses: Dict[str, float] = field(default_factory=dict)
    gradient_norms: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "total_loss": self.total_loss,
            "encoder_losses": self.encoder_losses,
            "gradient_norms": self.gradient_norms,
            "learning_rate": self.learning_rate,
        }


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation."""
    # Semantic fidelity (cosine similarity after encode-decode-encode)
    semantic_fidelity: Dict[str, float] = field(default_factory=dict)

    # Cross-modal retrieval
    recall_at_1: Dict[str, float] = field(default_factory=dict)
    recall_at_5: Dict[str, float] = field(default_factory=dict)
    recall_at_10: Dict[str, float] = field(default_factory=dict)

    # Alignment scores
    cross_modal_alignment: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_fidelity": self.semantic_fidelity,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "cross_modal_alignment": self.cross_modal_alignment,
        }


class MetricsCollector:
    """
    Collects and aggregates metrics during training and evaluation.
    """

    def __init__(self, encoder_names: List[str], output_dir: Optional[str] = None):
        """
        Initialize metrics collector.

        Args:
            encoder_names: Names of encoders being trained
            output_dir: Directory to save metrics (optional)
        """
        self.encoder_names = encoder_names
        self.output_dir = Path(output_dir) if output_dir else None

        # Training history
        self.epoch_metrics: List[EpochMetrics] = []

        # Evaluation history
        self.eval_metrics: List[EvaluationMetrics] = []

    def record_epoch(
        self,
        epoch: int,
        total_loss: float,
        encoder_losses: Dict[str, float],
        gradient_norms: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.0,
    ) -> EpochMetrics:
        """
        Record metrics for a training epoch.

        Args:
            epoch: Epoch number
            total_loss: Total training loss
            encoder_losses: Per-encoder losses
            gradient_norms: Per-encoder gradient norms (optional)
            learning_rate: Current learning rate

        Returns:
            EpochMetrics object
        """
        metrics = EpochMetrics(
            epoch=epoch,
            total_loss=total_loss,
            encoder_losses=encoder_losses,
            gradient_norms=gradient_norms or {},
            learning_rate=learning_rate,
        )
        self.epoch_metrics.append(metrics)
        return metrics

    def record_evaluation(self, metrics: EvaluationMetrics) -> None:
        """Record evaluation metrics."""
        self.eval_metrics.append(metrics)

    @staticmethod
    def compute_semantic_fidelity(
        original_vectors: torch.Tensor,
        reconstructed_vectors: torch.Tensor,
    ) -> float:
        """
        Compute semantic fidelity as cosine similarity.

        Args:
            original_vectors: Original semantic vectors (batch, dim)
            reconstructed_vectors: Re-encoded vectors after decode (batch, dim)

        Returns:
            Mean cosine similarity
        """
        original_norm = F.normalize(original_vectors, dim=1)
        reconstructed_norm = F.normalize(reconstructed_vectors, dim=1)

        # Compute cosine similarity
        similarities = torch.sum(original_norm * reconstructed_norm, dim=1)

        return similarities.mean().item()

    @staticmethod
    def compute_recall_at_k(
        query_vectors: torch.Tensor,
        key_vectors: torch.Tensor,
        k: int = 1,
    ) -> float:
        """
        Compute Recall@K for cross-modal retrieval.

        Args:
            query_vectors: Query vectors (batch, dim)
            key_vectors: Key vectors (batch, dim)
            k: Number of top results to consider

        Returns:
            Recall@K score (0-1)
        """
        query_norm = F.normalize(query_vectors, dim=1)
        key_norm = F.normalize(key_vectors, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(query_norm, key_norm.t())  # (batch, batch)

        # Get top-k indices for each query
        _, topk_indices = similarity.topk(k, dim=1)

        # Check if correct match is in top-k
        batch_size = query_vectors.shape[0]
        correct_indices = torch.arange(batch_size, device=query_vectors.device)
        correct_indices = correct_indices.unsqueeze(1).expand(-1, k)

        hits = (topk_indices == correct_indices).any(dim=1)

        return hits.float().mean().item()

    @staticmethod
    def compute_cross_modal_alignment(
        vectors_a: torch.Tensor,
        vectors_b: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute cross-modal alignment metrics.

        Args:
            vectors_a: Vectors from modality A (batch, dim)
            vectors_b: Vectors from modality B (batch, dim)

        Returns:
            Tuple of (mean alignment score, std alignment score)
        """
        a_norm = F.normalize(vectors_a, dim=1)
        b_norm = F.normalize(vectors_b, dim=1)

        # Diagonal similarities (same segment, different modality)
        diagonal_sim = torch.sum(a_norm * b_norm, dim=1)

        return diagonal_sim.mean().item(), diagonal_sim.std().item()

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training metrics.

        Returns:
            Dictionary with training summary
        """
        if not self.epoch_metrics:
            return {}

        losses = [m.total_loss for m in self.epoch_metrics]

        summary = {
            "num_epochs": len(self.epoch_metrics),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "best_loss": min(losses) if losses else None,
            "best_epoch": int(np.argmin(losses)) + 1 if losses else None,
            "loss_improvement": (losses[0] - losses[-1]) / losses[0] * 100 if losses and losses[0] > 0 else None,
        }

        # Per-encoder summaries
        for name in self.encoder_names:
            enc_losses = [m.encoder_losses.get(name, 0) for m in self.epoch_metrics]
            if enc_losses:
                summary[f"{name}_final_loss"] = enc_losses[-1]
                summary[f"{name}_best_loss"] = min(enc_losses)

        return summary

    def get_convergence_analysis(self) -> Dict[str, Any]:
        """
        Analyze training convergence.

        Returns:
            Dictionary with convergence analysis
        """
        if len(self.epoch_metrics) < 2:
            return {}

        losses = [m.total_loss for m in self.epoch_metrics]
        epochs = list(range(1, len(losses) + 1))

        # Compute moving average
        window = min(5, len(losses) // 2)
        if window > 0:
            smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        else:
            smoothed = losses

        # Check for convergence (loss change < 1% over last 5 epochs)
        if len(losses) >= 5:
            recent_losses = losses[-5:]
            loss_change = abs(recent_losses[-1] - recent_losses[0]) / max(recent_losses[0], 1e-8)
            converged = loss_change < 0.01
        else:
            converged = False

        return {
            "converged": converged,
            "final_smoothed_loss": float(smoothed[-1]) if len(smoothed) > 0 else None,
            "loss_trend": "decreasing" if len(smoothed) > 1 and smoothed[-1] < smoothed[0] else "stable_or_increasing",
        }

    def save(self, path: Optional[str] = None) -> None:
        """
        Save all collected metrics to JSON.

        Args:
            path: Output path (uses output_dir if not specified)
        """
        if path is None and self.output_dir:
            path = str(self.output_dir / "metrics.json")

        if path is None:
            return

        data = {
            "encoder_names": self.encoder_names,
            "epoch_metrics": [m.to_dict() for m in self.epoch_metrics],
            "eval_metrics": [m.to_dict() for m in self.eval_metrics],
            "training_summary": self.get_training_summary(),
            "convergence_analysis": self.get_convergence_analysis(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetricsCollector":
        """
        Load metrics from JSON file.

        Args:
            path: Path to metrics JSON file

        Returns:
            MetricsCollector with loaded data
        """
        with open(path, "r") as f:
            data = json.load(f)

        collector = cls(data.get("encoder_names", []))

        # Restore epoch metrics
        for m in data.get("epoch_metrics", []):
            collector.epoch_metrics.append(EpochMetrics(
                epoch=m["epoch"],
                total_loss=m["total_loss"],
                encoder_losses=m.get("encoder_losses", {}),
                gradient_norms=m.get("gradient_norms", {}),
                learning_rate=m.get("learning_rate", 0),
            ))

        # Restore eval metrics
        for m in data.get("eval_metrics", []):
            collector.eval_metrics.append(EvaluationMetrics(
                semantic_fidelity=m.get("semantic_fidelity", {}),
                recall_at_1=m.get("recall_at_1", {}),
                recall_at_5=m.get("recall_at_5", {}),
                recall_at_10=m.get("recall_at_10", {}),
                cross_modal_alignment=m.get("cross_modal_alignment", {}),
            ))

        return collector
