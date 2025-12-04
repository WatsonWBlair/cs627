"""
Visualization Module for Encoder/Decoder Evaluation

This module provides comprehensive visualization functions for analyzing
cross-modal encoder alignment and decoder reconstruction quality.

Key visualizations:
1. Confusion matrices for cross-modal retrieval
2. t-SNE/UMAP plots for semantic clustering
3. Similarity heatmaps for modality alignment
4. Training curves and loss plots
5. Performance bar charts across metrics
6. Reconstruction quality comparisons

Usage:
    from src.Evaluation.visualization import (
        plot_retrieval_confusion_matrix,
        plot_semantic_clustering,
        plot_similarity_heatmap,
        plot_training_curves,
        create_evaluation_dashboard
    )
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Try to import optional dependencies
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_retrieval_confusion_matrix(
    retrieval_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix for cross-modal retrieval results.
    
    Args:
        retrieval_results: Dict mapping "query_target" to recall metrics
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract modalities
    modality_pairs = list(retrieval_results.keys())
    modalities = sorted(set([m.split('_')[0] for m in modality_pairs] + 
                           [m.split('_')[1] for m in modality_pairs]))
    
    # Create retrieval matrix (R@1 scores)
    n_modalities = len(modalities)
    retrieval_matrix = np.zeros((n_modalities, n_modalities))
    
    for i, query_mod in enumerate(modalities):
        for j, target_mod in enumerate(modalities):
            if i == j:
                retrieval_matrix[i, j] = 1.0  # Perfect self-retrieval
            else:
                pair_key = f"{query_mod}_{target_mod}"
                if pair_key in retrieval_results:
                    retrieval_matrix[i, j] = retrieval_results[pair_key].get('R@1', 0.0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot R@1 heatmap
    sns.heatmap(
        retrieval_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        xticklabels=modalities,
        yticklabels=modalities,
        cbar_kws={'label': 'Recall@1'},
        ax=ax1
    )
    ax1.set_title('Cross-Modal Retrieval (R@1)')
    ax1.set_xlabel('Target Modality')
    ax1.set_ylabel('Query Modality')
    
    # Plot R@5 and R@10 as bar chart
    recall_data = {'R@1': [], 'R@5': [], 'R@10': []}
    pair_labels = []
    
    for pair, metrics in sorted(retrieval_results.items()):
        if not pair.startswith(pair.split('_')[0]):  # Skip self-retrieval
            continue
        pair_labels.append(pair.replace('_', '→'))
        for k in ['R@1', 'R@5', 'R@10']:
            recall_data[k].append(metrics.get(k, 0.0))
    
    x = np.arange(len(pair_labels))
    width = 0.25
    
    for i, (k, values) in enumerate(recall_data.items()):
        ax2.bar(x + i*width, values, width, label=k)
    
    ax2.set_xlabel('Modality Pair')
    ax2.set_ylabel('Recall Score')
    ax2.set_title('Retrieval Performance Comparison')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved retrieval confusion matrix to {save_path}")
    
    return fig


def plot_semantic_clustering(
    embeddings_dict: Dict[str, torch.Tensor],
    labels: Optional[List[str]] = None,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    max_samples: int = 1000
) -> plt.Figure:
    """
    Plot semantic clustering using dimensionality reduction.
    
    Args:
        embeddings_dict: Dict mapping modality name to embeddings tensor
        labels: Optional labels for coloring points
        method: 'tsne', 'umap', or 'pca'
        save_path: Optional path to save figure
        figsize: Figure size
        max_samples: Maximum samples to plot (for performance)
        
    Returns:
        Matplotlib figure
    """
    # Prepare data
    all_embeddings = []
    modality_labels = []
    
    for modality, emb in embeddings_dict.items():
        emb_np = emb.cpu().numpy() if torch.is_tensor(emb) else emb
        
        # Subsample if needed
        if len(emb_np) > max_samples:
            idx = np.random.choice(len(emb_np), max_samples, replace=False)
            emb_np = emb_np[idx]
        
        all_embeddings.append(emb_np)
        modality_labels.extend([modality] * len(emb_np))
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Apply dimensionality reduction
    print(f"Applying {method.upper()} to {len(all_embeddings)} samples...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(all_embeddings)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    if labels is not None and len(labels) == len(all_embeddings):
        # Plot with provided labels
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # Color by provided labels
        unique_labels = list(set(labels))
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax1.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.6,
                s=10
            )
        
        ax1.set_title(f'Semantic Clustering ({method.upper()}) - By Labels')
        
    else:
        # Plot by modality only
        ax1 = plt.subplot(111)
        ax2 = None
    
    # Color by modality
    unique_modalities = list(set(modality_labels))
    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(unique_modalities)))
    
    for i, modality in enumerate(unique_modalities):
        mask = np.array(modality_labels) == modality
        scatter = ax1.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=modality,
            alpha=0.6,
            s=20,
            marker=['o', 's', '^', 'D'][i % 4]
        )
    
    ax1.set_xlabel(f'{method.upper()} Component 1')
    ax1.set_ylabel(f'{method.upper()} Component 2')
    
    if ax2 is None:
        ax1.set_title(f'Semantic Space Visualization ({method.upper()})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax1.set_title(f'Semantic Clustering ({method.upper()}) - By Modality')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot modality distribution in second subplot
        modality_counts = {m: modality_labels.count(m) for m in unique_modalities}
        ax2.bar(range(len(modality_counts)), list(modality_counts.values()))
        ax2.set_xticks(range(len(modality_counts)))
        ax2.set_xticklabels(list(modality_counts.keys()), rotation=45)
        ax2.set_ylabel('Count')
        ax2.set_title('Modality Distribution')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved semantic clustering plot to {save_path}")
    
    return fig


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    modality_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot similarity heatmap between modalities.
    
    Args:
        similarity_matrix: (M, M) similarity matrix
        modality_names: List of modality names
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=modality_names,
        yticklabels=modality_names,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )
    
    ax.set_title('Cross-Modal Similarity Matrix')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity heatmap to {save_path}")
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot training curves (loss, metrics over epochs).
    
    Args:
        history: Dict mapping metric names to lists of values
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(history)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (metric_name, values) in enumerate(history.items()):
        ax = axes[i] if n_metrics > 1 else axes[0]
        
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, 'b-', label=metric_name)
        
        # Add trend line
        z = np.polyfit(epochs, values, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "r--", alpha=0.5, label='Trend')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def plot_performance_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot performance comparison across different models/configurations.
    
    Args:
        metrics_dict: Nested dict {model_name: {metric_name: value}}
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data
    models = list(metrics_dict.keys())
    all_metrics = set()
    for metrics in metrics_dict.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(list(all_metrics))
    
    # Create matrix for heatmap
    performance_matrix = np.zeros((len(models), len(all_metrics)))
    for i, model in enumerate(models):
        for j, metric in enumerate(all_metrics):
            performance_matrix[i, j] = metrics_dict[model].get(metric, 0.0)
    
    # Plot heatmap
    sns.heatmap(
        performance_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=all_metrics,
        yticklabels=models,
        cbar_kws={'label': 'Score'},
        ax=ax1
    )
    ax1.set_title('Performance Matrix')
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Model')
    
    # Plot radar chart for top metrics
    top_metrics = ['R@1', 'R@5', 'R@10', 'mean_similarity', 'alignment_score']
    available_metrics = [m for m in top_metrics if m in all_metrics]
    
    if available_metrics:
        angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        ax2 = plt.subplot(122, projection='polar')
        
        for model in models[:3]:  # Plot top 3 models
            values = [metrics_dict[model].get(m, 0.0) for m in available_metrics]
            values = values + [values[0]]  # Complete the circle
            ax2.plot(angles, values, 'o-', label=model)
            ax2.fill(angles, values, alpha=0.25)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(available_metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('Performance Radar Chart')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved performance comparison to {save_path}")
    
    return fig


def plot_decoder_reconstruction(
    original_texts: List[str],
    reconstructed_texts: List[str],
    similarity_scores: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    max_samples: int = 5
) -> plt.Figure:
    """
    Visualize decoder reconstruction quality with examples.
    
    Args:
        original_texts: Original input texts
        reconstructed_texts: Decoder output texts
        similarity_scores: Semantic similarity scores
        save_path: Optional path to save figure
        figsize: Figure size
        max_samples: Maximum samples to display
        
    Returns:
        Matplotlib figure
    """
    n_samples = min(max_samples, len(original_texts))
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    
    # Plot similarity distribution
    ax1 = plt.subplot(gs[0])
    ax1.hist(similarity_scores, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(similarity_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(similarity_scores):.3f}')
    ax1.set_xlabel('Semantic Similarity')
    ax1.set_ylabel('Count')
    ax1.set_title('Reconstruction Semantic Fidelity Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot reconstruction examples
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    
    # Create table data
    table_data = []
    for i in range(n_samples):
        # Truncate long texts
        orig = original_texts[i][:100] + "..." if len(original_texts[i]) > 100 else original_texts[i]
        recon = reconstructed_texts[i][:100] + "..." if len(reconstructed_texts[i]) > 100 else reconstructed_texts[i]
        sim = similarity_scores[i]
        
        table_data.append([f"Sample {i+1}", orig, recon, f"{sim:.3f}"])
    
    # Create table
    table = ax2.table(
        cellText=table_data,
        colLabels=['', 'Original', 'Reconstructed', 'Similarity'],
        cellLoc='left',
        loc='center',
        colWidths=[0.1, 0.4, 0.4, 0.1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color cells based on similarity
    for i in range(n_samples):
        sim_cell = table[(i+1, 3)]
        sim_value = similarity_scores[i]
        if sim_value > 0.9:
            sim_cell.set_facecolor('#90EE90')  # Light green
        elif sim_value > 0.7:
            sim_cell.set_facecolor('#FFFFE0')  # Light yellow
        else:
            sim_cell.set_facecolor('#FFB6C1')  # Light red
    
    plt.suptitle('Decoder Reconstruction Quality Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstruction visualization to {save_path}")
    
    return fig


def create_evaluation_dashboard(
    metrics: Dict[str, any],
    output_dir: str,
    title: str = "Cross-Modal Evaluation Dashboard"
) -> str:
    """
    Create comprehensive evaluation dashboard with all visualizations.
    
    Args:
        metrics: Complete metrics dictionary from evaluation
        output_dir: Directory to save figures
        title: Dashboard title
        
    Returns:
        Path to HTML dashboard
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    figures = []
    
    # 1. Retrieval confusion matrix
    if 'retrieval' in metrics:
        fig = plot_retrieval_confusion_matrix(
            metrics['retrieval'],
            save_path=output_dir / 'retrieval_confusion.png'
        )
        figures.append(('Retrieval Performance', 'retrieval_confusion.png'))
        plt.close(fig)
    
    # 2. Similarity heatmap
    if 'similarity_matrix' in metrics:
        fig = plot_similarity_heatmap(
            np.array(metrics['similarity_matrix']['matrix']),
            metrics['similarity_matrix']['modality_names'],
            save_path=output_dir / 'similarity_heatmap.png'
        )
        figures.append(('Similarity Matrix', 'similarity_heatmap.png'))
        plt.close(fig)
    
    # 3. Performance comparison
    if 'alignment_quality' in metrics:
        # Convert to format for plotting
        perf_dict = {
            modality: quality 
            for modality, quality in metrics['alignment_quality'].items()
        }
        fig = plot_performance_comparison(
            perf_dict,
            save_path=output_dir / 'performance_comparison.png'
        )
        figures.append(('Performance Comparison', 'performance_comparison.png'))
        plt.close(fig)
    
    # Generate HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
            .metrics {{ background: #f5f5f5; padding: 15px; margin: 20px 0; }}
            .metric-row {{ margin: 5px 0; }}
            .metric-label {{ font-weight: bold; display: inline-block; width: 200px; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="metrics">
            <h2>Summary Metrics</h2>
    """
    
    # Add summary metrics
    if 'retrieval' in metrics:
        all_r1 = [m.get('R@1', 0) for m in metrics['retrieval'].values()]
        avg_r1 = np.mean(all_r1) if all_r1 else 0
        html_content += f'<div class="metric-row"><span class="metric-label">Average R@1:</span>{avg_r1:.3f}</div>'
    
    html_content += """
        </div>
        <h2>Visualizations</h2>
    """
    
    # Add figures
    for title, filename in figures:
        html_content += f"""
        <div class="figure">
            <h3>{title}</h3>
            <img src="{filename}" alt="{title}">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML
    dashboard_path = output_dir / 'dashboard.html'
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created evaluation dashboard at {dashboard_path}")
    return str(dashboard_path)


if __name__ == "__main__":
    """Test visualization functions with sample data."""
    
    print("Testing visualization module...")
    
    # Create sample data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Sample embeddings
    embeddings = {
        'text': torch.randn(100, 1024),
        'audio': torch.randn(100, 1024),
        'image': torch.randn(100, 1024)
    }
    
    # Sample retrieval results
    retrieval_results = {
        'text_audio': {'R@1': 0.65, 'R@5': 0.82, 'R@10': 0.91},
        'text_image': {'R@1': 0.71, 'R@5': 0.85, 'R@10': 0.93},
        'audio_text': {'R@1': 0.62, 'R@5': 0.80, 'R@10': 0.89},
        'audio_image': {'R@1': 0.58, 'R@5': 0.76, 'R@10': 0.86},
        'image_text': {'R@1': 0.69, 'R@5': 0.84, 'R@10': 0.92},
        'image_audio': {'R@1': 0.56, 'R@5': 0.74, 'R@10': 0.85}
    }
    
    # Create output directory
    output_dir = Path('results/visualization_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test retrieval confusion matrix
    print("Plotting retrieval confusion matrix...")
    fig = plot_retrieval_confusion_matrix(
        retrieval_results,
        save_path=output_dir / 'test_retrieval.png'
    )
    plt.close(fig)
    
    # Test semantic clustering
    print("Plotting semantic clustering...")
    fig = plot_semantic_clustering(
        embeddings,
        method='pca',  # Use PCA for speed
        save_path=output_dir / 'test_clustering.png'
    )
    plt.close(fig)
    
    # Test similarity heatmap
    print("Plotting similarity heatmap...")
    sim_matrix = np.random.rand(3, 3)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(sim_matrix, 1.0)
    
    fig = plot_similarity_heatmap(
        sim_matrix,
        ['text', 'audio', 'image'],
        save_path=output_dir / 'test_similarity.png'
    )
    plt.close(fig)
    
    print(f"\nVisualization tests complete!")
    print(f"Figures saved to: {output_dir}")