"""
Main Evaluation Runner

Comprehensive evaluation script for trained cross-modal encoders.

This script performs:
1. Cross-modal retrieval evaluation (Recall@K)
2. Similarity matrix analysis
3. Alignment quality assessment
4. Optional downstream task evaluation (MultiBench)

Usage:
    # Basic evaluation on test split
    python scripts/run_evaluation.py

    # Evaluation with custom parameters
    python scripts/run_evaluation.py --split test --batch-size 32 --max-samples 200

    # Include downstream task evaluation
    python scripts/run_evaluation.py --downstream-task mosi_sentiment

    # Save results to custom directory
    python scripts/run_evaluation.py --output-dir results/my_evaluation/
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

import torch
from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
from src.Training.train_encoders import collate_fn_raw_video
from src.Evaluation import CrossModalEvaluator
from src.Evaluation.multibench_adapter import MOSISentimentTask


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained cross-modal encoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'valid', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--mosi-data-path',
        type=str,
        default='data/cmumosi/mosi/',
        help='Path to MOSI metadata'
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='data/cmumosi/audio/',
        help='Path to extracted audio files'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/cmumosi/frames/',
        help='Path to extracted video frames'
    )

    # Evaluation arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (None = all)'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help='K values for Recall@K computation'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation/',
        help='Directory to save results'
    )
    parser.add_argument(
        '--save-embeddings',
        action='store_true',
        help='Save encoded embeddings to disk'
    )

    # Downstream task arguments
    parser.add_argument(
        '--downstream-task',
        type=str,
        default=None,
        choices=['mosi_sentiment'],
        help='Downstream task to evaluate on (optional)'
    )
    parser.add_argument(
        '--task-epochs',
        type=int,
        default=20,
        help='Number of epochs for downstream task training'
    )

    # Encoder selection
    parser.add_argument(
        '--encoders',
        type=str,
        nargs='+',
        default=['text', 'audio_waveform', 'audio_tone', 'image'],
        help='Which encoders to evaluate'
    )

    return parser.parse_args()


def load_encoders(encoder_names):
    """
    Load trained encoders.

    Args:
        encoder_names: List of encoder names to load

    Returns:
        Dict mapping encoder name to loaded module
    """
    print("=" * 80)
    print("LOADING ENCODERS")
    print("=" * 80)
    print()

    encoders = {}

    for name in encoder_names:
        if name == 'text':
            print("Loading Text_to_Vec encoder...")
            encoders['text_base'] = Text_to_Vec()

        elif name == 'audio_waveform':
            print("Loading Audio_to_Vec (waveform) encoder...")
            encoders['audio_waveform'] = Audio_to_Vec()

        elif name == 'audio_tone':
            print("Loading Tone_to_Vec encoder...")
            # Import here to avoid circular dependency
            from src.Encoders.audio.tone_to_vec import Tone_to_Vec
            encoders['audio_tone'] = Tone_to_Vec()

        elif name == 'image':
            print("Loading Image_to_Vec encoder...")
            encoders['image_base'] = Image_to_Vec()

        else:
            print(f"[WARNING] Unknown encoder: {name}")

    print(f"\n[OK] Loaded {len(encoders)} encoders")
    print()

    return encoders


def load_dataset(args):
    """
    Load MOSI dataset.

    Args:
        args: Parsed command line arguments

    Returns:
        MOSIRawVideoDataset instance
    """
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    print()

    print(f"Split: {args.split}")
    print(f"MOSI data: {args.mosi_data_path}")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Video dir: {args.video_dir}")
    print()

    dataset = MOSIRawVideoDataset(
        split=args.split,
        mosi_data_path=args.mosi_data_path,
        audio_dir=args.audio_dir,
        video_dir=args.video_dir
    )

    # Set collate function
    dataset.collate_fn = collate_fn_raw_video

    print(f"[OK] Loaded {len(dataset)} samples")
    print()

    return dataset


def run_cross_modal_evaluation(encoders, dataset, args, output_dir):
    """
    Run cross-modal retrieval evaluation.

    Args:
        encoders: Dict of encoder modules
        dataset: Dataset to evaluate on
        args: Command line arguments
        output_dir: Directory to save results

    Returns:
        Evaluation results dict
    """
    print("=" * 80)
    print("CROSS-MODAL RETRIEVAL EVALUATION")
    print("=" * 80)
    print()

    # Create evaluator
    evaluator = CrossModalEvaluator(encoders)

    # Run evaluation
    results_path = output_dir / "cross_modal_metrics.json"
    results = evaluator.evaluate(
        dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        k_values=args.k_values,
        save_results=str(results_path)
    )

    # Optionally save embeddings
    if args.save_embeddings:
        print("\n[INFO] Saving encoded embeddings...")
        embeddings = evaluator.encode_dataset(dataset, args.batch_size, args.max_samples)

        embeddings_path = output_dir / "embeddings.pt"
        torch.save(embeddings, embeddings_path)
        print(f"[OK] Saved embeddings to: {embeddings_path}")

    return results


def run_downstream_task(encoders, dataset, args, output_dir):
    """
    Run downstream task evaluation (MultiBench).

    Args:
        encoders: Dict of encoder modules
        dataset: Dataset to evaluate on
        args: Command line arguments
        output_dir: Directory to save results

    Returns:
        Task evaluation results dict
    """
    print("=" * 80)
    print(f"DOWNSTREAM TASK: {args.downstream_task.upper()}")
    print("=" * 80)
    print()

    if args.downstream_task == "mosi_sentiment":
        # Create sentiment task adapter
        adapter = MOSISentimentTask(
            encoders=encoders,
            fusion_method="concat",
            hidden_dim=128,
            freeze_encoders=True
        )

        # TODO: Split dataset into train/val
        # For now, just report that this would be trained
        print("[INFO] Downstream task evaluation:")
        print("  Task: MOSI Sentiment (7-class classification)")
        print("  Encoders: Frozen")
        print("  Task head: 2-layer MLP")
        print()
        print("[TODO] Implement train/val split and task training")
        print("  Use adapter.train_on_task(train_loader, val_loader)")

        return {"status": "not_implemented"}

    else:
        print(f"[ERROR] Unknown downstream task: {args.downstream_task}")
        return {"status": "error"}


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"eval_{args.split}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("CROSS-MODAL ENCODER EVALUATION")
    print("=" * 80)
    print()
    print(f"Timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")
    print(f"Evaluating on: {args.split} split")
    print()

    # Save configuration
    config = vars(args)
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Saved configuration to: {config_path}")
    print()

    # 1. Load encoders
    encoders = load_encoders(args.encoders)

    # 2. Load dataset
    dataset = load_dataset(args)

    # 3. Run cross-modal evaluation
    cross_modal_results = run_cross_modal_evaluation(encoders, dataset, args, output_dir)

    # 4. Optionally run downstream task
    if args.downstream_task:
        downstream_results = run_downstream_task(encoders, dataset, args, output_dir)

    # 5. Generate summary report
    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("Files generated:")
    print(f"  - config.json (evaluation configuration)")
    print(f"  - cross_modal_metrics.json (retrieval metrics)")
    if args.save_embeddings:
        print(f"  - embeddings.pt (encoded embeddings)")
    print()

    # Print key metrics
    print("Key Results:")
    print()

    # Average retrieval scores
    retrieval = cross_modal_results.get("retrieval", {})
    if retrieval:
        r1_scores = [metrics["R@1"] for metrics in retrieval.values()]
        r5_scores = [metrics["R@5"] for metrics in retrieval.values()]
        r10_scores = [metrics["R@10"] for metrics in retrieval.values()]

        avg_r1 = sum(r1_scores) / len(r1_scores)
        avg_r5 = sum(r5_scores) / len(r5_scores)
        avg_r10 = sum(r10_scores) / len(r10_scores)

        print(f"  Average Recall@1:  {avg_r1:.3f}")
        print(f"  Average Recall@5:  {avg_r5:.3f}")
        print(f"  Average Recall@10: {avg_r10:.3f}")
        print()

    # Alignment quality
    alignment = cross_modal_results.get("alignment_quality", {})
    if alignment:
        print("  Alignment Quality:")
        for modality, quality in alignment.items():
            score = quality.get("alignment_score", 0.0)
            print(f"    {modality:20s} {score:.3f}")
        print()

    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
