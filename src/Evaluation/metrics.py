"""
Evaluation Metrics for Cross-Modal Encoder Alignment

This module provides core metric computation functions for evaluating
the quality of multimodal encoder alignment to the shared semantic vector space.

Key Metrics:
1. Recall@K - Cross-modal retrieval accuracy
2. Cosine Similarity Matrix - Pairwise similarity between modalities
3. Alignment Quality - Variance and consistency of cross-modal embeddings
4. Modality Gap - Distance between modality-specific embedding distributions
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist


def compute_recall_at_k(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Recall@K for cross-modal retrieval.

    For each query, rank all targets by similarity and check if the
    ground-truth target (same index) appears in the top-K results.

    Args:
        query_embeddings: (N, D) tensor of query embeddings
        target_embeddings: (N, D) tensor of target embeddings
        k_values: List of K values to compute recall for

    Returns:
        Dict mapping "R@{k}" to recall value (0.0 to 1.0)

    Example:
        >>> text_emb = torch.randn(100, 1024)
        >>> audio_emb = torch.randn(100, 1024)
        >>> metrics = compute_recall_at_k(text_emb, audio_emb, [1, 5, 10])
        >>> print(f"Text→Audio R@1: {metrics['R@1']:.3f}")
    """
    # Normalize embeddings
    query_norm = F.normalize(query_embeddings, p=2, dim=1)
    target_norm = F.normalize(target_embeddings, p=2, dim=1)

    # Compute similarity matrix (N x N)
    # similarity[i, j] = cosine_sim(query[i], target[j])
    similarity = torch.mm(query_norm, target_norm.t())

    # Get sorted indices (descending similarity)
    # ranks[i] contains indices of targets sorted by similarity to query[i]
    _, ranks = torch.sort(similarity, dim=1, descending=True)

    # Ground truth: for query i, target i is the correct match
    num_samples = query_embeddings.shape[0]
    ground_truth = torch.arange(num_samples, device=query_embeddings.device)

    # Compute recall for each K
    recall_dict = {}
    for k in k_values:
        # For each query, check if ground truth appears in top-K
        top_k = ranks[:, :k]

        # Check if ground_truth[i] appears in top_k[i]
        correct = (top_k == ground_truth.unsqueeze(1)).any(dim=1)

        recall = correct.float().mean().item()
        recall_dict[f"R@{k}"] = recall

    return recall_dict


def compute_cosine_similarity_matrix(
    embeddings_dict: Dict[str, torch.Tensor],
    aggregate: str = "mean"
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise cosine similarity matrix between all modalities.

    Args:
        embeddings_dict: Dict mapping modality name to (N, D) embeddings
        aggregate: How to aggregate similarities ("mean", "median", "std")

    Returns:
        similarity_matrix: (M, M) array of pairwise similarities
        modality_names: List of modality names (ordering for matrix)

    Example:
        >>> emb = {
        ...     "text": torch.randn(100, 1024),
        ...     "audio": torch.randn(100, 1024),
        ...     "image": torch.randn(100, 1024)
        ... }
        >>> matrix, names = compute_cosine_similarity_matrix(emb)
        >>> # matrix[0, 1] = similarity between text and audio
    """
    modality_names = sorted(embeddings_dict.keys())
    num_modalities = len(modality_names)

    similarity_matrix = np.zeros((num_modalities, num_modalities))

    for i, mod1 in enumerate(modality_names):
        for j, mod2 in enumerate(modality_names):
            emb1 = embeddings_dict[mod1]
            emb2 = embeddings_dict[mod2]

            # Normalize embeddings
            emb1_norm = F.normalize(emb1, p=2, dim=1)
            emb2_norm = F.normalize(emb2, p=2, dim=1)

            # Compute pairwise similarities
            # For aligned data, similarity[i, i] should be high
            similarities = torch.sum(emb1_norm * emb2_norm, dim=1)

            # Aggregate
            if aggregate == "mean":
                similarity_matrix[i, j] = similarities.mean().item()
            elif aggregate == "median":
                similarity_matrix[i, j] = similarities.median().item()
            elif aggregate == "std":
                similarity_matrix[i, j] = similarities.std().item()
            else:
                raise ValueError(f"Unknown aggregate: {aggregate}")

    return similarity_matrix, modality_names


def compute_alignment_quality(
    embeddings_dict: Dict[str, torch.Tensor],
    reference_modality: str = "text"
) -> Dict[str, float]:
    """
    Compute alignment quality metrics for each modality vs. reference.

    Measures how consistently each modality aligns to the reference modality
    using variance of within-pair distances.

    Args:
        embeddings_dict: Dict mapping modality name to (N, D) embeddings
        reference_modality: Which modality to use as ground truth

    Returns:
        Dict mapping modality name to alignment quality score
        (lower variance = better alignment)

    Example:
        >>> emb = {"text": ..., "audio": ..., "image": ...}
        >>> quality = compute_alignment_quality(emb, reference_modality="text")
        >>> print(f"Audio alignment quality: {quality['audio']:.3f}")
    """
    if reference_modality not in embeddings_dict:
        raise ValueError(f"Reference modality '{reference_modality}' not found")

    ref_emb = embeddings_dict[reference_modality]
    ref_norm = F.normalize(ref_emb, p=2, dim=1)

    quality_scores = {}

    for modality, emb in embeddings_dict.items():
        if modality == reference_modality:
            continue

        # Normalize embeddings
        emb_norm = F.normalize(emb, p=2, dim=1)

        # Compute within-pair cosine distances
        # distance[i] = 1 - cos_sim(ref[i], emb[i])
        cosine_sim = torch.sum(ref_norm * emb_norm, dim=1)
        distances = 1.0 - cosine_sim

        # Lower variance = more consistent alignment
        variance = distances.var().item()
        mean_distance = distances.mean().item()

        quality_scores[modality] = {
            "mean_distance": mean_distance,
            "variance": variance,
            "alignment_score": 1.0 / (1.0 + variance)  # Higher is better
        }

    return quality_scores


def compute_modality_gap(
    embeddings_dict: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute modality gap - distance between modality-specific distributions.

    Uses maximum mean discrepancy (MMD) to measure distribution shift between
    modalities. Lower gap = better alignment to shared semantic space.

    Args:
        embeddings_dict: Dict mapping modality name to (N, D) embeddings

    Returns:
        Dict mapping modality pair to MMD distance

    Reference:
        Gretton et al. "A Kernel Two-Sample Test" (2012)
    """
    modality_names = sorted(embeddings_dict.keys())
    gaps = {}

    for i, mod1 in enumerate(modality_names):
        for j, mod2 in enumerate(modality_names[i+1:], start=i+1):
            emb1 = embeddings_dict[mod1].cpu().numpy()
            emb2 = embeddings_dict[mod2].cpu().numpy()

            # Compute pairwise distances
            XX = cdist(emb1, emb1, metric='euclidean')
            YY = cdist(emb2, emb2, metric='euclidean')
            XY = cdist(emb1, emb2, metric='euclidean')

            # RBF kernel bandwidth (median heuristic)
            bandwidth = np.median(XY)

            # Compute RBF kernel
            K_XX = np.exp(-XX / (2 * bandwidth**2))
            K_YY = np.exp(-YY / (2 * bandwidth**2))
            K_XY = np.exp(-XY / (2 * bandwidth**2))

            # Maximum Mean Discrepancy (MMD)
            n = emb1.shape[0]
            m = emb2.shape[0]

            mmd = (K_XX.sum() / (n * n) +
                   K_YY.sum() / (m * m) -
                   2 * K_XY.sum() / (n * m))

            gaps[f"{mod1}_{mod2}"] = mmd

    return gaps


def compute_intra_modal_variance(
    embeddings: torch.Tensor
) -> Dict[str, float]:
    """
    Compute variance statistics within a single modality's embeddings.

    Useful for detecting collapsed representations (very low variance)
    or overly dispersed embeddings (very high variance).

    Args:
        embeddings: (N, D) tensor of embeddings

    Returns:
        Dict with variance statistics
    """
    # Normalize first
    emb_norm = F.normalize(embeddings, p=2, dim=1)

    # Compute centroid
    centroid = emb_norm.mean(dim=0, keepdim=True)
    centroid_norm = F.normalize(centroid, p=2, dim=1)

    # Distance from centroid
    distances = 1.0 - torch.sum(emb_norm * centroid_norm, dim=1)

    return {
        "mean_dist_from_centroid": distances.mean().item(),
        "std_dist_from_centroid": distances.std().item(),
        "min_dist_from_centroid": distances.min().item(),
        "max_dist_from_centroid": distances.max().item()
    }


def compute_cross_modal_metrics(
    embeddings_dict: Dict[str, torch.Tensor],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, any]:
    """
    Compute comprehensive cross-modal evaluation metrics.

    This is the main entry point for evaluation. Computes all metrics
    including retrieval accuracy, similarity matrices, alignment quality,
    and modality gaps.

    Args:
        embeddings_dict: Dict mapping modality name to (N, D) embeddings
        k_values: K values for Recall@K computation

    Returns:
        Nested dict with all computed metrics

    Example:
        >>> from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
        >>> # ... encode dataset ...
        >>> embeddings = {
        ...     "text": text_embeddings,
        ...     "audio": audio_embeddings,
        ...     "image": image_embeddings
        ... }
        >>> metrics = compute_cross_modal_metrics(embeddings)
        >>> print(f"Text→Audio R@1: {metrics['retrieval']['text_audio']['R@1']:.3f}")
    """
    modality_names = sorted(embeddings_dict.keys())

    results = {
        "retrieval": {},
        "similarity_matrix": {},
        "alignment_quality": {},
        "modality_gap": {},
        "intra_modal_variance": {}
    }

    # 1. Cross-modal retrieval (all pairs)
    for i, query_mod in enumerate(modality_names):
        for j, target_mod in enumerate(modality_names):
            if i == j:
                continue  # Skip same-modality retrieval

            pair_name = f"{query_mod}_{target_mod}"
            recall = compute_recall_at_k(
                embeddings_dict[query_mod],
                embeddings_dict[target_mod],
                k_values
            )
            results["retrieval"][pair_name] = recall

    # 2. Cosine similarity matrix
    sim_matrix, names = compute_cosine_similarity_matrix(embeddings_dict)
    results["similarity_matrix"]["matrix"] = sim_matrix.tolist()
    results["similarity_matrix"]["modality_names"] = names

    # 3. Alignment quality (using text as reference if available)
    reference = "text" if "text" in modality_names else modality_names[0]
    results["alignment_quality"] = compute_alignment_quality(
        embeddings_dict,
        reference_modality=reference
    )

    # 4. Modality gap (MMD between distributions)
    results["modality_gap"] = compute_modality_gap(embeddings_dict)

    # 5. Intra-modal variance
    for modality, emb in embeddings_dict.items():
        results["intra_modal_variance"][modality] = compute_intra_modal_variance(emb)

    return results


def print_metrics_summary(metrics: Dict[str, any]):
    """
    Pretty-print metrics summary for quick inspection.

    Args:
        metrics: Output from compute_cross_modal_metrics()
    """
    print("=" * 80)
    print("CROSS-MODAL EVALUATION METRICS")
    print("=" * 80)
    print()

    # Retrieval results
    print("CROSS-MODAL RETRIEVAL (Recall@K)")
    print("-" * 80)
    for pair, recall in sorted(metrics["retrieval"].items()):
        r1 = recall.get("R@1", 0.0)
        r5 = recall.get("R@5", 0.0)
        r10 = recall.get("R@10", 0.0)
        print(f"  {pair:20s}  R@1: {r1:.3f}  R@5: {r5:.3f}  R@10: {r10:.3f}")
    print()

    # Similarity matrix
    print("COSINE SIMILARITY MATRIX (Mean)")
    print("-" * 80)
    matrix = np.array(metrics["similarity_matrix"]["matrix"])
    names = metrics["similarity_matrix"]["modality_names"]

    # Print header
    print(f"  {'':12s}", end="")
    for name in names:
        print(f"{name:>10s}", end="")
    print()

    # Print matrix
    for i, name in enumerate(names):
        print(f"  {name:12s}", end="")
        for j in range(len(names)):
            print(f"{matrix[i, j]:>10.3f}", end="")
        print()
    print()

    # Alignment quality
    print("ALIGNMENT QUALITY (vs. reference)")
    print("-" * 80)
    for modality, quality in sorted(metrics["alignment_quality"].items()):
        mean_dist = quality["mean_distance"]
        variance = quality["variance"]
        score = quality["alignment_score"]
        print(f"  {modality:12s}  Distance: {mean_dist:.3f}  Variance: {variance:.4f}  Score: {score:.3f}")
    print()

    # Modality gap
    print("MODALITY GAP (MMD)")
    print("-" * 80)
    for pair, mmd in sorted(metrics["modality_gap"].items()):
        print(f"  {pair:20s}  MMD: {mmd:.4f}")
    print()

    print("=" * 80)


if __name__ == "__main__":
    # Example usage with random data
    print("Testing metrics computation with random embeddings...")
    print()

    # Generate random embeddings for 3 modalities
    torch.manual_seed(42)
    embeddings = {
        "text": torch.randn(50, 1024),
        "audio": torch.randn(50, 1024),
        "image": torch.randn(50, 1024)
    }

    # Compute all metrics
    metrics = compute_cross_modal_metrics(embeddings, k_values=[1, 5, 10])

    # Print summary
    print_metrics_summary(metrics)

    print("\nExample completed successfully!")
