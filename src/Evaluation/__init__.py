"""
Evaluation infrastructure for cross-modal encoder alignment.

This module provides tools for evaluating the quality of encoder alignment
to the shared semantic vector space.

Key Components:
- metrics: Core metric computation functions (Recall@K, similarity, MMD)
- CrossModalEvaluator: High-level evaluator for datasets
- MultiBenchAdapter: Adapter for MultiBench benchmark suite

Usage:
    from src.Evaluation import CrossModalEvaluator, compute_cross_modal_metrics
    from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec

    encoders = {"text": Text_to_Vec(), "audio": Audio_to_Vec()}
    evaluator = CrossModalEvaluator(encoders)
    results = evaluator.evaluate(dataset)
"""

from .encoder_metrics import (
    compute_recall_at_k,
    compute_cosine_similarity_matrix,
    compute_alignment_quality,
    compute_modality_gap,
    compute_cross_modal_metrics,
    print_metrics_summary
)

from .evaluators import CrossModalEvaluator

__all__ = [
    "compute_recall_at_k",
    "compute_cosine_similarity_matrix",
    "compute_alignment_quality",
    "compute_modality_gap",
    "compute_cross_modal_metrics",
    "print_metrics_summary",
    "CrossModalEvaluator"
]
