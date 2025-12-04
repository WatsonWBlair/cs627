"""
Standalone Decoder Evaluation Script

Command-line interface for evaluating trained decoder modules using the MOSI dataset.

Usage:
    # Evaluate text decoder on test split
    python scripts/evaluate_decoder.py --split test --batch-size 32

    # Evaluate on validation split with custom output directory
    python scripts/evaluate_decoder.py --split val --output results/my_eval/

    # Compare multiple checkpoints
    python scripts/evaluate_decoder.py --compare checkpoint1.pth checkpoint2.pth --split test

    # Limit number of samples for quick testing
    python scripts/evaluate_decoder.py --split test --max-samples 100
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.Encoders import Text_to_Vec
from src.Decoders import Vec_to_Text
from src.Evaluation.decoder_evaluator import DecoderEvaluator
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained decoder modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test split with default settings
  python scripts/evaluate_decoder.py --split test

  # Evaluate on validation split with larger batch size
  python scripts/evaluate_decoder.py --split val --batch-size 64

  # Compare multiple checkpoints
  python scripts/evaluate_decoder.py --compare checkpoint1.pth checkpoint2.pth --split test

  # Quick test on small subset
  python scripts/evaluate_decoder.py --split test --max-samples 50 --batch-size 16
        """
    )

    # Dataset arguments
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset split to evaluate on (default: test)'
    )

    parser.add_argument(
        '--mosi-data-path',
        type=str,
        default='data/cmumosi/mosi/',
        help='Path to MOSI metadata (default: data/cmumosi/mosi/)'
    )

    parser.add_argument(
        '--audio-dir',
        type=str,
        default='data/cmumosi/audio/',
        help='Path to audio files (default: data/cmumosi/audio/)'
    )

    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/cmumosi/frames/',
        help='Path to video frames (default: data/cmumosi/frames/)'
    )

    # Evaluation arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/decoder_evaluation/',
        help='Output directory for results (default: results/decoder_evaluation/)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )

    parser.add_argument(
        '--save-reconstructions',
        action='store_true',
        help='Save reconstructed texts to output file'
    )

    # Checkpoint comparison mode
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Compare multiple decoder checkpoints (space-separated paths)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    print("=" * 80)
    print("DECODER EVALUATION SCRIPT")
    print("=" * 80)
    print()

    # Determine device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Configuration:")
    print(f"  Split:           {args.split}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Max samples:     {args.max_samples or 'all'}")
    print(f"  Device:          {device}")
    print(f"  Output dir:      {args.output}")
    print()

    # Load dataset
    print("[1/3] Loading dataset...")
    print()

    try:
        dataset = MOSIRawVideoDataset(
            split=args.split,
            mosi_data_path=args.mosi_data_path,
            audio_dir=args.audio_dir,
            video_dir=args.video_dir,
            required_modalities=['text']  # Text-only evaluation for decoder
        )

        print(f"  [OK] Loaded {len(dataset)} samples from {args.split} split")
        print()

    except FileNotFoundError as e:
        print(f"  [FAIL] Dataset not found: {e}")
        print()
        print("Please run data preparation scripts first:")
        print("  1. python scripts/data_wrangling/download_all_mosi_videos.py")
        print("  2. python scripts/data_wrangling/extract_test_segments.py")
        print()
        return 1

    # Load models
    print("[2/3] Loading models...")
    print()

    decoder = Vec_to_Text()
    encoder = Text_to_Vec()

    print(f"  [OK] Decoder: {type(decoder).__name__}")
    print(f"  [OK] Encoder: {type(encoder).__name__}")
    print()

    # Create evaluator
    evaluator = DecoderEvaluator(
        decoder=decoder,
        encoder=encoder,
        output_dir=args.output,
        device=device,
        verbose=not args.quiet
    )

    # Run evaluation
    print("[3/3] Running evaluation...")
    print()

    if args.compare:
        # Checkpoint comparison mode
        print(f"Comparing {len(args.compare)} checkpoints:")
        for i, checkpoint in enumerate(args.compare, 1):
            print(f"  [{i}] {checkpoint}")
        print()

        # Extract test samples
        test_texts = []
        num_samples = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)

        for i in range(num_samples):
            sample = dataset[i]
            test_texts.append(sample['text'])

        # Run comparison
        results = evaluator.compare_checkpoints(
            checkpoint_paths=args.compare,
            test_data=test_texts,
            batch_size=args.batch_size
        )

        # Print comparison summary
        print()
        print("=" * 80)
        print("CHECKPOINT COMPARISON SUMMARY")
        print("=" * 80)
        print()

        for checkpoint_name, metrics in results.items():
            print(f"{checkpoint_name}:")
            print(f"  BLEU:             {metrics['bleu']['bleu']:.4f}")
            print(f"  ROUGE-L:          {metrics['rouge']['rougeL']:.4f}")
            print(f"  Semantic Fidelity: {metrics['semantic_fidelity']['mean_similarity']:.4f}")
            print(f"  Combined Quality: {metrics['reconstruction_quality']:.4f}")
            print()

    else:
        # Standard evaluation mode
        results = evaluator.evaluate_from_dataset(
            dataset=dataset,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            save_results=True,
            save_reconstructions=args.save_reconstructions
        )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print()

    if not args.compare:
        metrics = results['metrics']
        print("Final Metrics:")
        print(f"  BLEU:             {metrics['bleu']['bleu']:.4f}")
        print(f"  ROUGE-L:          {metrics['rouge']['rougeL']:.4f}")
        print(f"  Semantic Fidelity: {metrics['semantic_fidelity']['mean_similarity']:.4f}")
        print(f"  Combined Quality: {metrics['reconstruction_quality']:.4f}")
        print()

    print(f"Results saved to: {args.output}")
    print()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
