#!/usr/bin/env python
"""
Decoder Training Visualization Script

Generates publication-quality figures from decoder training history.

Usage:
    python scripts/generate_decoder_figures.py
    python scripts/generate_decoder_figures.py --output-dir figures/decoders
    python scripts/generate_decoder_figures.py --publication --dpi 300
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate decoder training figures")
    parser.add_argument('--checkpoint-dir', type=str, default='DecoderCheckpoints',
                        help='Directory containing decoder checkpoints')
    parser.add_argument('--output-dir', type=str, default='figures/decoders',
                        help='Directory to save figures')
    parser.add_argument('--publication', action='store_true',
                        help='Generate publication-quality figures')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format')
    return parser.parse_args()


def load_training_history(checkpoint_dir: Path) -> dict:
    """Load all decoder training histories."""
    histories = {}

    for decoder_dir in checkpoint_dir.iterdir():
        if decoder_dir.is_dir():
            history_file = decoder_dir / 'training_history.json'
            if history_file.exists():
                with open(history_file) as f:
                    histories[decoder_dir.name] = json.load(f)
                print(f"  Loaded {decoder_dir.name} decoder history ({len(histories[decoder_dir.name].get('history', []))} epochs)")

    return histories


def plot_decoder_training_curves(histories: dict, output_dir: Path, args):
    """Generate training curves for all decoders."""
    print("\nGenerating training curves...")

    for decoder_name, data in histories.items():
        history = data.get('history', [])
        if not history:
            continue

        epochs = [h['epoch'] for h in history]

        # Extract metrics
        metrics = {}
        for h in history:
            for key, value in h['metrics'].items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

        # Create figure with subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        colors = ['#2E86AB', '#A23B72', '#F18F01']

        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i]

            # Plot metric
            ax.plot(epochs, values, '-o', color=colors[i % len(colors)],
                   linewidth=2, markersize=4, label=metric_name)

            # Add trend line
            z = np.polyfit(epochs, values, 2)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), '--', color=colors[i % len(colors)],
                   alpha=0.5, linewidth=1)

            # Mark best value
            if 'loss' in metric_name.lower():
                best_idx = np.argmin(values)
                best_label = 'Min'
            else:
                best_idx = np.argmax(values)
                best_label = 'Max'

            ax.axhline(y=values[best_idx], color='gray', linestyle=':', alpha=0.5)
            ax.scatter([epochs[best_idx]], [values[best_idx]],
                      color='red', s=100, zorder=5, marker='*')
            ax.annotate(f'{best_label}: {values[best_idx]:.4f}',
                       xy=(epochs[best_idx], values[best_idx]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color='red')

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{decoder_name.title()} Decoder Training', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        output_path = output_dir / f'{decoder_name}_training_curves.{args.format}'
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)


def plot_combined_decoder_metrics(histories: dict, output_dir: Path, args):
    """Generate combined visualization comparing all decoders."""
    print("\nGenerating combined metrics visualization...")

    if len(histories) < 1:
        print("  No decoder histories found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'text': '#2E86AB', 'audio': '#A23B72', 'image': '#F18F01'}

    for decoder_name, data in histories.items():
        history = data.get('history', [])
        if not history:
            continue

        epochs = [h['epoch'] for h in history]
        color = colors.get(decoder_name, '#333333')

        # Plot 1: Loss curves
        ax = axes[0, 0]
        loss_key = None
        for key in history[0]['metrics'].keys():
            if 'loss' in key.lower():
                loss_key = key
                break

        if loss_key:
            values = [h['metrics'][loss_key] for h in history]
            ax.plot(epochs, values, '-o', color=color, linewidth=2,
                   markersize=4, label=f'{decoder_name}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Semantic Similarity
        ax = axes[0, 1]
        if 'semantic_similarity' in history[0]['metrics']:
            values = [h['metrics']['semantic_similarity'] for h in history]
            ax.plot(epochs, values, '-o', color=color, linewidth=2,
                   markersize=4, label=f'{decoder_name}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Semantic Similarity')
        ax.set_title('Semantic Fidelity')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target (0.85)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Semantic Drift
        ax = axes[1, 0]
        if 'semantic_drift' in history[0]['metrics']:
            values = [h['metrics']['semantic_drift'] for h in history]
            ax.plot(epochs, values, '-o', color=color, linewidth=2,
                   markersize=4, label=f'{decoder_name}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Semantic Drift')
        ax.set_title('Semantic Drift (Lower is Better)')
        ax.set_ylim(0, 0.5)
        ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='Target (<0.15)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Training duration per epoch
        ax = axes[1, 1]
        durations = [h['duration'] for h in history]
        ax.bar(epochs, durations, color=color, alpha=0.7, label=decoder_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Duration (seconds)')
        ax.set_title('Training Time per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Decoder Training Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'decoder_training_summary.{args.format}'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_final_metrics_comparison(histories: dict, output_dir: Path, args):
    """Generate bar chart comparing final metrics across decoders."""
    print("\nGenerating final metrics comparison...")

    if len(histories) < 1:
        print("  No decoder histories found")
        return

    # Collect final metrics
    decoder_names = []
    final_metrics = {}

    for decoder_name, data in histories.items():
        history = data.get('history', [])
        if not history:
            continue

        decoder_names.append(decoder_name)
        final = history[-1]['metrics']

        for key, value in final.items():
            if key not in final_metrics:
                final_metrics[key] = []
            final_metrics[key].append(value)

    if not decoder_names:
        return

    # Create figure
    n_metrics = len(final_metrics)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(decoder_names))
    width = 0.8 / n_metrics

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for i, (metric_name, values) in enumerate(final_metrics.items()):
        offset = (i - n_metrics/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name.replace('_', ' ').title(),
                     color=colors[i % len(colors)])

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Decoder')
    ax.set_ylabel('Metric Value')
    ax.set_title('Final Training Metrics by Decoder')
    ax.set_xticks(x)
    ax.set_xticklabels([n.title() for n in decoder_names])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / f'decoder_final_metrics.{args.format}'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_hyperparameters_table(histories: dict, output_dir: Path, args):
    """Generate a table visualization of hyperparameters."""
    print("\nGenerating hyperparameters table...")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Collect hyperparameters
    table_data = []
    headers = ['Decoder', 'Batch Size', 'Learning Rate', 'Epochs', 'Training Approach', 'Best Loss']

    for decoder_name, data in histories.items():
        hp = data.get('hyperparameters', {})
        best_loss = data.get('best_loss', 'N/A')

        row = [
            decoder_name.title(),
            str(hp.get('batch_size', 'N/A')),
            str(hp.get('learning_rate', 'N/A')),
            str(hp.get('epochs', 'N/A')),
            hp.get('training_approach', 'N/A').replace('_', ' ').title(),
            f"{best_loss:.4f}" if isinstance(best_loss, float) else str(best_loss)
        ]
        table_data.append(row)

    if not table_data:
        print("  No data for table")
        plt.close(fig)
        return

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.12, 0.14, 0.1, 0.28, 0.14]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Decoder Training Hyperparameters', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = output_dir / f'decoder_hyperparameters.{args.format}'
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    args = parse_args()

    print("="*60)
    print("DECODER TRAINING VISUALIZATION")
    print("="*60)

    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.publication:
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}, Format: {args.format}")

    # Load training histories
    print("\nLoading training histories...")
    histories = load_training_history(checkpoint_dir)

    if not histories:
        print("\n[ERROR] No training histories found!")
        print(f"Expected files at: {checkpoint_dir}/*/training_history.json")
        return 1

    print(f"\nFound {len(histories)} decoder(s): {list(histories.keys())}")

    # Generate visualizations
    plot_decoder_training_curves(histories, output_dir, args)
    plot_combined_decoder_metrics(histories, output_dir, args)
    plot_final_metrics_comparison(histories, output_dir, args)
    plot_hyperparameters_table(histories, output_dir, args)

    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Figures saved to: {output_dir}")
    print("\nGenerated files:")
    for fig_file in sorted(output_dir.glob(f'*.{args.format}')):
        print(f"  - {fig_file.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
