#!/usr/bin/env python
"""
Ablation Study Visualization Script

Generates visualizations from ablation study results to analyze
the impact of different adapter configurations.

Usage:
    # Visualize latest ablation results
    python scripts/visualize_ablation.py

    # Visualize specific results file
    python scripts/visualize_ablation.py --results Results/ablation/ablation_results_*.json

    # Save figures to custom directory
    python scripts/visualize_ablation.py --output figures/ablation/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')


def load_ablation_results(results_path: str) -> Dict:
    """Load ablation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def find_latest_results(results_dir: str = 'Results/ablation') -> Optional[str]:
    """Find the most recent ablation results file."""
    pattern = os.path.join(results_dir, 'ablation_results_*.json')
    files = glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def plot_config_comparison(results, save_path: Optional[str] = None) -> plt.Figure:
    """
    Bar chart comparing validation loss across configurations.
    """
    # Handle both list and dict formats
    if isinstance(results, list):
        configs = results
    else:
        configs = results.get('results', [])

    if not configs:
        print("No configuration results found")
        return None

    names = [c['config']['name'] for c in configs]
    val_losses = [c.get('best_val_loss', c.get('metrics', {}).get('best_val_loss', float('inf'))) for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), val_losses, color=sns.color_palette("viridis", len(names)))

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title('Ablation Study: Configuration Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')

    # Add value labels on bars
    for bar, val in zip(bars, val_losses):
        if val != float('inf'):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_training_curves(results, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training curves for all configurations.
    """
    # Handle both list and dict formats
    if isinstance(results, list):
        configs = results
    else:
        configs = results.get('results', [])

    if not configs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = sns.color_palette("husl", len(configs))

    # Training loss
    ax1 = axes[0]
    for i, config in enumerate(configs):
        history = config.get('history', {})
        train_loss = history.get('train_loss', [])
        if train_loss:
            ax1.plot(train_loss, label=config['config']['name'], color=colors[i], alpha=0.8)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Validation loss
    ax2 = axes[1]
    for i, config in enumerate(configs):
        history = config.get('history', {})
        val_loss = history.get('val_loss', [])
        if val_loss:
            ax2.plot(val_loss, label=config['config']['name'], color=colors[i], alpha=0.8)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_parameter_heatmap(results, save_path: Optional[str] = None) -> plt.Figure:
    """
    Heatmap showing relationship between parameters and performance.
    """
    # Handle both list and dict formats
    if isinstance(results, list):
        configs = results
    else:
        configs = results.get('results', [])

    if not configs:
        return None

    # Extract parameters and metrics
    data = []
    for config in configs:
        cfg = config.get('config', config)
        # Handle both nested metrics and flat structure
        val_loss = config.get('best_val_loss', config.get('metrics', {}).get('best_val_loss', float('inf')))
        data.append({
            'hidden_size': cfg.get('hidden_size', 512),
            'num_layers': cfg.get('num_layers', 2),
            'dropout': cfg.get('dropout', 0.1),
            'activation': cfg.get('activation', 'relu'),
            'val_loss': val_loss
        })

    df = pd.DataFrame(data)

    # Filter out infinite values
    df = df[df['val_loss'] != float('inf')]

    if len(df) < 2:
        print("Not enough valid results for heatmap")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hidden size vs Layers heatmap
    ax1 = axes[0]
    try:
        pivot1 = df.pivot_table(values='val_loss', index='hidden_size',
                                columns='num_layers', aggfunc='mean')
        sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='viridis_r', ax=ax1)
        ax1.set_title('Val Loss: Hidden Size vs Layers', fontsize=12, fontweight='bold')
    except Exception as e:
        ax1.text(0.5, 0.5, f'Insufficient data\nfor heatmap', ha='center', va='center')
        ax1.set_title('Hidden Size vs Layers', fontsize=12)

    # Dropout effect
    ax2 = axes[1]
    if 'dropout' in df.columns and df['dropout'].nunique() > 1:
        dropout_perf = df.groupby('dropout')['val_loss'].mean()
        ax2.bar(dropout_perf.index.astype(str), dropout_perf.values, color='steelblue')
        ax2.set_xlabel('Dropout Rate', fontsize=11)
        ax2.set_ylabel('Avg Validation Loss', fontsize=11)
        ax2.set_title('Effect of Dropout on Performance', fontsize=12, fontweight='bold')
    else:
        # Show activation comparison instead
        if 'activation' in df.columns:
            act_perf = df.groupby('activation')['val_loss'].mean()
            ax2.bar(act_perf.index, act_perf.values, color='coral')
            ax2.set_xlabel('Activation Function', fontsize=11)
            ax2.set_ylabel('Avg Validation Loss', fontsize=11)
            ax2.set_title('Effect of Activation on Performance', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_best_config_summary(results, save_path: Optional[str] = None) -> plt.Figure:
    """
    Summary visualization of the best configuration.
    """
    # Handle both list and dict formats
    if isinstance(results, list):
        configs = results
    else:
        configs = results.get('results', [])

    if not configs:
        return None

    # Helper to get val_loss from either structure
    def get_val_loss(c):
        return c.get('best_val_loss', c.get('metrics', {}).get('best_val_loss', float('inf')))

    # Find best config
    valid_configs = [c for c in configs if get_val_loss(c) != float('inf')]
    if not valid_configs:
        print("No valid configurations found")
        return None

    best = min(valid_configs, key=get_val_loss)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Config details (text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    cfg = best.get('config', best)
    best_metrics = best.get('best_metrics', best.get('metrics', {}))
    best_val_loss = get_val_loss(best)
    best_epoch = best_metrics.get('best_epoch', best.get('epochs_trained', 'N/A'))
    training_time = best_metrics.get('training_time', best.get('training_time', 0))

    text = f"""Best Configuration: {cfg.get('name', 'N/A')}

Hidden Size: {cfg.get('hidden_size', 'N/A')}
Num Layers: {cfg.get('num_layers', 'N/A')}
Activation: {cfg.get('activation', 'N/A')}
Dropout: {cfg.get('dropout', 'N/A')}

Performance:
Best Val Loss: {best_val_loss:.4f}
Best Epoch: {best_epoch}
Training Time: {training_time:.1f}s"""

    ax1.text(0.1, 0.9, text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax1.set_title('Best Configuration Summary', fontsize=12, fontweight='bold')

    # Training curve for best config
    ax2 = fig.add_subplot(gs[0, 1])
    history = best.get('history', {})
    if history.get('train_loss') and history.get('val_loss'):
        epochs = range(1, len(history['train_loss']) + 1)
        ax2.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax2.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        if isinstance(best_epoch, int):
            ax2.axvline(x=best_epoch, color='g', linestyle='--',
                       label=f"Best Epoch ({best_epoch})")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'Training Curve: {cfg.get("name", "Best")}', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Comparison with all configs
    ax3 = fig.add_subplot(gs[1, :])
    names = [c.get('config', c).get('name', f'Config {i}') for i, c in enumerate(configs)]
    losses = [get_val_loss(c) for c in configs]

    # Replace inf with max for visualization
    max_valid = max([l for l in losses if l != float('inf')] or [1])
    losses_viz = [l if l != float('inf') else max_valid * 1.2 for l in losses]

    best_name = cfg.get('name', '')
    colors = ['green' if c.get('config', c).get('name', '') == best_name else 'steelblue'
              for c in configs]
    bars = ax3.bar(range(len(names)), losses_viz, color=colors)
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Best Validation Loss')
    ax3.set_title('All Configurations (Best in Green)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_ablation_report(results, output_dir: str = 'Results/ablation/figures'):
    """
    Generate all ablation visualizations and save to output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating Ablation Study Visualizations")
    print("=" * 60)

    # Generate all plots
    print("\n1. Configuration comparison...")
    plot_config_comparison(results, save_path=output_path / 'config_comparison.png')

    print("\n2. Training curves...")
    plot_training_curves(results, save_path=output_path / 'training_curves.png')

    print("\n3. Parameter heatmap...")
    plot_parameter_heatmap(results, save_path=output_path / 'parameter_heatmap.png')

    print("\n4. Best config summary...")
    plot_best_config_summary(results, save_path=output_path / 'best_config_summary.png')

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualize ablation study results')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to ablation results JSON file')
    parser.add_argument('--output', type=str, default='Results/ablation/figures',
                       help='Output directory for figures')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    args = parser.parse_args()

    # Find results file
    if args.results:
        results_path = args.results
    else:
        results_path = find_latest_results()

    if not results_path or not os.path.exists(results_path):
        print("No ablation results found!")
        print("Run ablation study first: make.bat ablation-local")
        return

    print(f"Loading results from: {results_path}")
    results = load_ablation_results(results_path)

    # Generate report
    create_ablation_report(results, args.output)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
