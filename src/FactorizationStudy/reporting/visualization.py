"""
Visualization module for Factorization Study results.

Generates:
- Dimension comparison bar charts
- Adapter configuration heatmaps
- Training curves by configuration
- Resource usage charts
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Matplotlib configuration for non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.FactorizationStudy.config.study_config import StudyConfig
from src.FactorizationStudy.runner.experiment import ExperimentResults


class Visualizer:
    """
    Creates visualizations from factorization study results.
    """

    def __init__(
        self,
        study_config: StudyConfig,
        results: Dict[str, ExperimentResults],
    ):
        """
        Initialize visualizer.

        Args:
            study_config: Study configuration
            results: Dictionary mapping experiment IDs to results
        """
        self.study_config = study_config
        self.results = results
        self.output_dir = study_config.get_study_dir() / "reports" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            256: "#2ecc71",   # Green
            512: "#3498db",   # Blue
            1024: "#9b59b6",  # Purple
            2048: "#e74c3c",  # Red
        }

    def generate_all_plots(self) -> None:
        """Generate all visualization plots."""
        print(f"\n[Visualizer] Generating plots to {self.output_dir}")

        self.plot_dimension_comparison()
        self.plot_adapter_heatmap()
        self.plot_training_curves()
        self.plot_resource_usage()

        print(f"[Visualizer] All plots generated")

    def plot_dimension_comparison(self) -> str:
        """
        Generate bar chart comparing performance across semantic dimensions.

        Returns:
            Path to generated plot
        """
        completed_results = [r for r in self.results.values() if r.completed]
        if not completed_results:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Group by dimension
        dim_losses = {}
        dim_times = {}

        for dim in self.study_config.semantic_dims:
            dim_results = [r for r in completed_results if r.config.get("semantic_dim") == dim]
            if dim_results:
                dim_losses[dim] = [r.best_loss for r in dim_results]
                dim_times[dim] = [
                    r.resource_summary.total_training_time_seconds / 3600  # Convert to hours
                    for r in dim_results
                    if r.resource_summary
                ]

        dims = list(dim_losses.keys())
        x = np.arange(len(dims))
        width = 0.6

        # Plot 1: Loss by dimension
        ax1 = axes[0]
        means = [np.mean(dim_losses[d]) for d in dims]
        stds = [np.std(dim_losses[d]) for d in dims]
        colors = [self.colors.get(d, "#95a5a6") for d in dims]

        bars = ax1.bar(x, means, width, yerr=stds, capsize=5, color=colors, edgecolor='black')
        ax1.set_xlabel('Semantic Dimension', fontsize=12)
        ax1.set_ylabel('Best Loss', fontsize=12)
        ax1.set_title('Loss by Semantic Dimension', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(d) for d in dims])
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Training time by dimension
        ax2 = axes[1]
        if dim_times:
            means = [np.mean(dim_times[d]) if dim_times.get(d) else 0 for d in dims]
            stds = [np.std(dim_times[d]) if dim_times.get(d) else 0 for d in dims]

            bars = ax2.bar(x, means, width, yerr=stds, capsize=5, color=colors, edgecolor='black')
            ax2.set_xlabel('Semantic Dimension', fontsize=12)
            ax2.set_ylabel('Training Time (hours)', fontsize=12)
            ax2.set_title('Training Time by Dimension', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([str(d) for d in dims])
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        path = self.output_dir / "dimension_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[Visualizer] Generated dimension comparison: {path}")
        return str(path)

    def plot_adapter_heatmap(self) -> str:
        """
        Generate heatmap of performance across adapter configurations.

        Returns:
            Path to generated plot
        """
        completed_results = [r for r in self.results.values() if r.completed]
        if not completed_results:
            return ""

        hidden_sizes = sorted(self.study_config.hidden_sizes)
        hidden_layers = sorted(self.study_config.hidden_layers)

        # Build loss matrix
        loss_matrix = np.zeros((len(hidden_layers), len(hidden_sizes)))
        count_matrix = np.zeros((len(hidden_layers), len(hidden_sizes)))

        for result in completed_results:
            hs = result.config.get("encoder_adapter", {}).get("hidden_size")
            hl = result.config.get("encoder_adapter", {}).get("hidden_layers")

            if hs in hidden_sizes and hl in hidden_layers:
                i = hidden_layers.index(hl)
                j = hidden_sizes.index(hs)
                loss_matrix[i, j] += result.best_loss
                count_matrix[i, j] += 1

        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_matrix = np.where(count_matrix > 0, loss_matrix / count_matrix, np.nan)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(avg_matrix, cmap='RdYlGn_r', aspect='auto')

        # Labels
        ax.set_xticks(np.arange(len(hidden_sizes)))
        ax.set_yticks(np.arange(len(hidden_layers)))
        ax.set_xticklabels([str(s) for s in hidden_sizes])
        ax.set_yticklabels([str(l) for l in hidden_layers])
        ax.set_xlabel('Hidden Size', fontsize=12)
        ax.set_ylabel('Hidden Layers', fontsize=12)
        ax.set_title('Mean Loss by Adapter Configuration\n(lower is better)', fontsize=14, fontweight='bold')

        # Annotate cells
        for i in range(len(hidden_layers)):
            for j in range(len(hidden_sizes)):
                if not np.isnan(avg_matrix[i, j]):
                    text = ax.text(j, i, f'{avg_matrix[i, j]:.3f}',
                                   ha='center', va='center', color='black', fontsize=10)

        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Mean Loss', rotation=-90, va='bottom', fontsize=11)

        plt.tight_layout()

        path = self.output_dir / "adapter_heatmap.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[Visualizer] Generated adapter heatmap: {path}")
        return str(path)

    def plot_training_curves(self) -> str:
        """
        Generate training curves for select experiments.

        Returns:
            Path to generated plot
        """
        completed_results = [r for r in self.results.values() if r.completed and r.epoch_metrics]
        if not completed_results:
            return ""

        # Select representative experiments (best from each dimension)
        representative = []
        for dim in self.study_config.semantic_dims:
            dim_results = [r for r in completed_results if r.config.get("semantic_dim") == dim]
            if dim_results:
                best = min(dim_results, key=lambda r: r.best_loss)
                representative.append(best)

        if not representative:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        for result in representative:
            dim = result.config.get("semantic_dim")
            epochs = [m.epoch for m in result.epoch_metrics]
            losses = [m.total_loss for m in result.epoch_metrics]

            color = self.colors.get(dim, "#95a5a6")
            ax.plot(epochs, losses, linewidth=2, marker='o', markersize=3,
                    color=color, label=f'{dim}D')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Total Loss', fontsize=12)
        ax.set_title('Training Curves by Semantic Dimension\n(Best Configuration per Dimension)',
                     fontsize=14, fontweight='bold')
        ax.legend(title='Semantic Dim', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        path = self.output_dir / "training_curves.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[Visualizer] Generated training curves: {path}")
        return str(path)

    def plot_resource_usage(self) -> str:
        """
        Generate resource usage visualization.

        Returns:
            Path to generated plot
        """
        completed_results = [
            r for r in self.results.values()
            if r.completed and r.resource_summary
        ]
        if not completed_results:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Group by dimension
        dims = self.study_config.semantic_dims
        dim_gpu = {d: [] for d in dims}
        dim_time = {d: [] for d in dims}

        for result in completed_results:
            dim = result.config.get("semantic_dim")
            if dim in dims:
                dim_gpu[dim].append(result.resource_summary.peak_gpu_memory_mb)
                dim_time[dim].append(result.resource_summary.avg_epoch_time_seconds)

        x = np.arange(len(dims))
        width = 0.6

        # Plot 1: GPU Memory
        ax1 = axes[0]
        means = [np.mean(dim_gpu[d]) if dim_gpu[d] else 0 for d in dims]
        stds = [np.std(dim_gpu[d]) if dim_gpu[d] else 0 for d in dims]
        colors = [self.colors.get(d, "#95a5a6") for d in dims]

        ax1.bar(x, means, width, yerr=stds, capsize=5, color=colors, edgecolor='black')
        ax1.set_xlabel('Semantic Dimension', fontsize=12)
        ax1.set_ylabel('Peak GPU Memory (MB)', fontsize=12)
        ax1.set_title('GPU Memory Usage by Dimension', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(d) for d in dims])
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Epoch Time
        ax2 = axes[1]
        means = [np.mean(dim_time[d]) if dim_time[d] else 0 for d in dims]
        stds = [np.std(dim_time[d]) if dim_time[d] else 0 for d in dims]

        ax2.bar(x, means, width, yerr=stds, capsize=5, color=colors, edgecolor='black')
        ax2.set_xlabel('Semantic Dimension', fontsize=12)
        ax2.set_ylabel('Average Epoch Time (seconds)', fontsize=12)
        ax2.set_title('Training Speed by Dimension', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(d) for d in dims])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        path = self.output_dir / "resource_usage.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[Visualizer] Generated resource usage plot: {path}")
        return str(path)
