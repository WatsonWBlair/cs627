#!/usr/bin/env python3
"""
Compare trained weights between different training runs.

Lists all training runs in CandidateWeights/, compares metrics between runs,
and allows promoting best weights to OptimalWeights/.

Usage:
    python scripts/cloud/compare_weights.py                     # List all runs
    python scripts/cloud/compare_weights.py --compare run1 run2 # Compare two runs
    python scripts/cloud/compare_weights.py --best              # Show best run
    python scripts/cloud/compare_weights.py --promote run1      # Promote to OptimalWeights/
"""

import os
import sys
import json
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CANDIDATE_WEIGHTS_DIR = PROJECT_ROOT / "CandidateWeights"
OPTIMAL_WEIGHTS_DIR = PROJECT_ROOT / "OptimalWeights"


def get_training_runs() -> List[Dict]:
    """Get all training runs from CandidateWeights directory."""
    runs = []

    if not CANDIDATE_WEIGHTS_DIR.exists():
        return runs

    for run_dir in CANDIDATE_WEIGHTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        run_info = {
            'name': run_dir.name,
            'path': run_dir,
            'timestamp': None,
            'metrics': None,
            'weight_files': [],
            'has_encoder_checkpoints': False,
            'has_decoder_checkpoints': False,
        }

        # Parse timestamp from directory name (format: hostname_YYYYMMDD_HHMMSS)
        parts = run_dir.name.rsplit('_', 2)
        if len(parts) >= 3:
            try:
                date_str = f"{parts[-2]}_{parts[-1]}"
                run_info['timestamp'] = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
            except ValueError:
                pass

        # Check for metrics file
        metrics_file = run_dir / "training_reports" / "training_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    run_info['metrics'] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Count weight files
        weights_dir = run_dir / "OptimalWeights"
        if weights_dir.exists():
            run_info['weight_files'] = list(weights_dir.glob("*.pth"))

        # Check for checkpoints
        encoder_ckpt = run_dir / "EncoderCheckpoints"
        decoder_ckpt = run_dir / "DecoderCheckpoints"
        run_info['has_encoder_checkpoints'] = encoder_ckpt.exists() and any(encoder_ckpt.iterdir()) if encoder_ckpt.exists() else False
        run_info['has_decoder_checkpoints'] = decoder_ckpt.exists() and any(decoder_ckpt.iterdir()) if decoder_ckpt.exists() else False

        runs.append(run_info)

    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x['timestamp'] or datetime.min, reverse=True)
    return runs


def format_metrics(metrics: Optional[Dict]) -> str:
    """Format metrics for display."""
    if not metrics:
        return "No metrics available"

    lines = []

    # Final loss
    if 'final_loss' in metrics:
        lines.append(f"  Final Loss: {metrics['final_loss']:.4f}")

    # Individual encoder losses
    if 'encoder_losses' in metrics:
        for encoder, loss in metrics['encoder_losses'].items():
            lines.append(f"  {encoder}: {loss:.4f}")

    # Training info
    if 'epochs_completed' in metrics:
        lines.append(f"  Epochs: {metrics['epochs_completed']}")

    if 'training_duration' in metrics:
        lines.append(f"  Duration: {metrics['training_duration']}")

    return '\n'.join(lines) if lines else "No metrics available"


def list_runs(runs: List[Dict]) -> None:
    """List all training runs."""
    if not runs:
        print("No training runs found in CandidateWeights/")
        print("Run training and download results to populate this directory.")
        return

    print("=" * 80)
    print("Available Training Runs")
    print("=" * 80)
    print()

    for i, run in enumerate(runs, 1):
        timestamp_str = run['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if run['timestamp'] else "Unknown"
        weight_count = len(run['weight_files'])

        print(f"[{i}] {run['name']}")
        print(f"    Timestamp: {timestamp_str}")
        print(f"    Weights: {weight_count} adapter files")

        checkpoints = []
        if run['has_encoder_checkpoints']:
            checkpoints.append("Encoder")
        if run['has_decoder_checkpoints']:
            checkpoints.append("Decoder")
        if checkpoints:
            print(f"    Checkpoints: {', '.join(checkpoints)}")

        if run['metrics']:
            print(f"    Metrics:")
            print(format_metrics(run['metrics']))

        print()


def compare_runs(runs: List[Dict], run1_name: str, run2_name: str) -> None:
    """Compare metrics between two runs."""
    run1 = next((r for r in runs if r['name'] == run1_name), None)
    run2 = next((r for r in runs if r['name'] == run2_name), None)

    if not run1:
        print(f"Error: Run '{run1_name}' not found")
        return
    if not run2:
        print(f"Error: Run '{run2_name}' not found")
        return

    print("=" * 80)
    print(f"Comparing: {run1_name} vs {run2_name}")
    print("=" * 80)
    print()

    # Compare timestamps
    t1 = run1['timestamp'].strftime("%Y-%m-%d %H:%M") if run1['timestamp'] else "Unknown"
    t2 = run2['timestamp'].strftime("%Y-%m-%d %H:%M") if run2['timestamp'] else "Unknown"
    print(f"{'Metric':<25} {'Run 1':<25} {'Run 2':<25}")
    print("-" * 80)
    print(f"{'Timestamp':<25} {t1:<25} {t2:<25}")
    print(f"{'Weight Files':<25} {len(run1['weight_files']):<25} {len(run2['weight_files']):<25}")

    # Compare metrics
    m1 = run1['metrics'] or {}
    m2 = run2['metrics'] or {}

    if 'final_loss' in m1 or 'final_loss' in m2:
        l1 = f"{m1.get('final_loss', 'N/A'):.4f}" if isinstance(m1.get('final_loss'), (int, float)) else "N/A"
        l2 = f"{m2.get('final_loss', 'N/A'):.4f}" if isinstance(m2.get('final_loss'), (int, float)) else "N/A"

        # Indicate which is better
        better = ""
        if isinstance(m1.get('final_loss'), (int, float)) and isinstance(m2.get('final_loss'), (int, float)):
            if m1['final_loss'] < m2['final_loss']:
                better = " <-- BETTER"
                l1 += better
            elif m2['final_loss'] < m1['final_loss']:
                better = " <-- BETTER"
                l2 += better

        print(f"{'Final Loss':<25} {l1:<25} {l2:<25}")

    # Compare encoder losses
    all_encoders = set(m1.get('encoder_losses', {}).keys()) | set(m2.get('encoder_losses', {}).keys())
    for encoder in sorted(all_encoders):
        l1 = m1.get('encoder_losses', {}).get(encoder)
        l2 = m2.get('encoder_losses', {}).get(encoder)

        l1_str = f"{l1:.4f}" if l1 is not None else "N/A"
        l2_str = f"{l2:.4f}" if l2 is not None else "N/A"

        if l1 is not None and l2 is not None:
            if l1 < l2:
                l1_str += " <-- BETTER"
            elif l2 < l1:
                l2_str += " <-- BETTER"

        print(f"  {encoder:<23} {l1_str:<25} {l2_str:<25}")

    print()


def find_best_run(runs: List[Dict]) -> Optional[Dict]:
    """Find the run with the lowest final loss."""
    best_run = None
    best_loss = float('inf')

    for run in runs:
        if run['metrics'] and 'final_loss' in run['metrics']:
            loss = run['metrics']['final_loss']
            if loss < best_loss:
                best_loss = loss
                best_run = run

    return best_run


def show_best(runs: List[Dict]) -> None:
    """Show the best training run."""
    best = find_best_run(runs)

    if not best:
        print("No runs with metrics found.")
        print("Ensure training_metrics.json is present in training_reports/")
        return

    print("=" * 80)
    print("Best Training Run (lowest final loss)")
    print("=" * 80)
    print()
    print(f"Run: {best['name']}")
    if best['timestamp']:
        print(f"Timestamp: {best['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Weight Files: {len(best['weight_files'])}")
    print()
    print("Metrics:")
    print(format_metrics(best['metrics']))
    print()
    print("To promote these weights to production:")
    print(f"  python scripts/cloud/compare_weights.py --promote {best['name']}")
    print()


def promote_run(runs: List[Dict], run_name: str, force: bool = False) -> None:
    """Promote a run's weights to OptimalWeights/."""
    run = next((r for r in runs if r['name'] == run_name), None)

    if not run:
        print(f"Error: Run '{run_name}' not found")
        return

    if not run['weight_files']:
        print(f"Error: No weight files found in {run_name}")
        return

    source_dir = run['path'] / "OptimalWeights"

    print("=" * 80)
    print(f"Promoting weights from: {run_name}")
    print("=" * 80)
    print()
    print(f"Source: {source_dir}")
    print(f"Target: {OPTIMAL_WEIGHTS_DIR}")
    print()
    print("Files to copy:")
    for wf in run['weight_files']:
        print(f"  - {wf.name}")
    print()

    if not force:
        response = input("Proceed with promotion? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    # Create OptimalWeights if needed
    OPTIMAL_WEIGHTS_DIR.mkdir(exist_ok=True)

    # Copy files
    copied = 0
    for wf in run['weight_files']:
        target = OPTIMAL_WEIGHTS_DIR / wf.name
        shutil.copy2(wf, target)
        print(f"Copied: {wf.name}")
        copied += 1

    print()
    print(f"Successfully promoted {copied} weight files to OptimalWeights/")
    print()

    # Also copy metrics if available
    metrics_source = run['path'] / "training_reports" / "training_metrics.json"
    if metrics_source.exists():
        metrics_target = OPTIMAL_WEIGHTS_DIR / "training_metrics.json"
        shutil.copy2(metrics_source, metrics_target)
        print(f"Also copied training_metrics.json for reference")


def main():
    parser = argparse.ArgumentParser(
        description="Compare trained weights between different training runs"
    )
    parser.add_argument(
        '--compare', nargs=2, metavar=('RUN1', 'RUN2'),
        help='Compare two specific runs'
    )
    parser.add_argument(
        '--best', action='store_true',
        help='Show the best run based on final loss'
    )
    parser.add_argument(
        '--promote', metavar='RUN',
        help='Promote a run\'s weights to OptimalWeights/'
    )
    parser.add_argument(
        '--force', '-f', action='store_true',
        help='Skip confirmation when promoting'
    )

    args = parser.parse_args()

    # Get all runs
    runs = get_training_runs()

    if args.compare:
        compare_runs(runs, args.compare[0], args.compare[1])
    elif args.best:
        show_best(runs)
    elif args.promote:
        promote_run(runs, args.promote, args.force)
    else:
        list_runs(runs)


if __name__ == "__main__":
    main()
