"""
Decoder Evaluation Pipeline

Comprehensive evaluation framework for decoder modules (Vec_to_Text, Vec_to_Audio, Vec_to_Image).
Provides end-to-end evaluation workflow including data loading, metric computation, and result saving.

Usage:
    from src.Evaluation.decoder_evaluator import DecoderEvaluator
    from src.Encoders import Text_to_Vec
    from src.Decoders import Vec_to_Text

    evaluator = DecoderEvaluator(
        decoder=Vec_to_Text(),
        encoder=Text_to_Vec(),
        output_dir='results/decoder_evaluation/'
    )

    results = evaluator.evaluate(
        test_texts=['example text 1', 'example text 2', ...],
        batch_size=32
    )
"""

import os
import sys
import json
import torch
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.Evaluation.decoder_metrics import (
    evaluate_text_decoder,
    print_decoder_metrics_summary,
    save_decoder_metrics,
    compute_bleu_scores,
    compute_rouge_scores,
    compute_semantic_fidelity
)


class DecoderEvaluator:
    """
    Evaluation pipeline for decoder modules.

    Manages the full evaluation workflow:
    1. Load decoder and encoder modules
    2. Process test data in batches
    3. Compute comprehensive metrics
    4. Save results to disk
    5. Generate evaluation report

    Args:
        decoder: Decoder module (Vec_to_Text, Vec_to_Audio, etc.)
        encoder: Corresponding encoder module (Text_to_Vec, Audio_to_Vec, etc.)
        output_dir: Directory to save evaluation results
        device: 'cuda' or 'cpu'
        verbose: Print detailed progress messages
    """

    def __init__(
        self,
        decoder,
        encoder,
        output_dir: str = 'results/decoder_evaluation/',
        device: str = None,
        verbose: bool = True
    ):
        self.decoder = decoder
        self.encoder = encoder
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Move models to device
        self.decoder.to(self.device)
        self.encoder.to(self.device)

        # Set models to eval mode
        self.decoder.eval()
        self.encoder.eval()

        if self.verbose:
            print("=" * 80)
            print("DECODER EVALUATOR INITIALIZED")
            print("=" * 80)
            print(f"Decoder: {type(self.decoder).__name__}")
            print(f"Encoder: {type(self.encoder).__name__}")
            print(f"Device: {self.device}")
            print(f"Output directory: {self.output_dir}")
            print("=" * 80)
            print()

    def evaluate(
        self,
        test_data: List[str],
        batch_size: int = 32,
        save_results: bool = True,
        save_reconstructions: bool = False
    ) -> Dict[str, any]:
        """
        Run comprehensive evaluation on test data.

        Args:
            test_data: List of test samples (text strings)
            batch_size: Batch size for processing
            save_results: Whether to save results to disk
            save_reconstructions: Whether to save reconstructed outputs

        Returns:
            Dict with all evaluation metrics and metadata
        """
        if self.verbose:
            print("=" * 80)
            print("STARTING EVALUATION")
            print("=" * 80)
            print(f"Test samples: {len(test_data)}")
            print(f"Batch size: {batch_size}")
            print()

        # Run evaluation
        start_time = datetime.now()

        metrics = evaluate_text_decoder(
            decoder=self.decoder,
            encoder=self.encoder,
            test_texts=test_data,
            device=self.device,
            batch_size=batch_size,
            verbose=self.verbose
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Add metadata
        results = {
            'metrics': metrics,
            'metadata': {
                'decoder_type': type(self.decoder).__name__,
                'encoder_type': type(self.encoder).__name__,
                'num_samples': len(test_data),
                'batch_size': batch_size,
                'device': self.device,
                'evaluation_time': duration,
                'timestamp': start_time.isoformat()
            }
        }

        # Print summary
        if self.verbose:
            print()
            print_decoder_metrics_summary(metrics, show_examples=5)
            print()
            print(f"Evaluation completed in {duration:.1f} seconds")
            print()

        # Save results
        if save_results:
            self._save_results(results, save_reconstructions)

        return results

    def _save_results(self, results: Dict[str, any], save_reconstructions: bool):
        """Save evaluation results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics JSON
        metrics_path = os.path.join(self.output_dir, f"metrics_{timestamp}.json")
        save_decoder_metrics(
            results['metrics'],
            metrics_path,
            include_reconstructions=save_reconstructions
        )

        # Save full results (including metadata)
        full_results_path = os.path.join(self.output_dir, f"full_results_{timestamp}.json")
        serializable_results = {
            'metadata': results['metadata'],
            'bleu': results['metrics']['bleu'],
            'rouge': results['metrics']['rouge'],
            'semantic_fidelity': results['metrics']['semantic_fidelity'],
            'reconstruction_quality': results['metrics']['reconstruction_quality']
        }

        if save_reconstructions and 'reconstructed_texts' in results['metrics']:
            serializable_results['reconstructed_texts'] = results['metrics']['reconstructed_texts']

        with open(full_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        if self.verbose:
            print(f"[OK] Results saved:")
            print(f"     Metrics: {metrics_path}")
            print(f"     Full results: {full_results_path}")
            print()

        # Save text report
        report_path = os.path.join(self.output_dir, f"report_{timestamp}.txt")
        self._save_text_report(results, report_path)

        if self.verbose:
            print(f"[OK] Text report: {report_path}")
            print()

    def _save_text_report(self, results: Dict[str, any], output_path: str):
        """Generate and save human-readable text report."""
        metrics = results['metrics']
        metadata = results['metadata']

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DECODER EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Metadata
            f.write("EVALUATION METADATA\n")
            f.write("-" * 80 + "\n")
            f.write(f"Timestamp:       {metadata['timestamp']}\n")
            f.write(f"Decoder:         {metadata['decoder_type']}\n")
            f.write(f"Encoder:         {metadata['encoder_type']}\n")
            f.write(f"Test samples:    {metadata['num_samples']}\n")
            f.write(f"Batch size:      {metadata['batch_size']}\n")
            f.write(f"Device:          {metadata['device']}\n")
            f.write(f"Duration:        {metadata['evaluation_time']:.1f}s\n")
            f.write("\n")

            # Overall quality
            f.write("OVERALL RECONSTRUCTION QUALITY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Combined Score:  {metrics['reconstruction_quality']:.4f}\n")
            f.write("\n")

            # BLEU
            f.write("BLEU SCORES (Precision-based)\n")
            f.write("-" * 80 + "\n")
            bleu = metrics['bleu']
            f.write(f"BLEU (overall):  {bleu['bleu']:.4f}\n")
            f.write(f"BLEU-1 (1-gram): {bleu['bleu_1']:.4f}\n")
            f.write(f"BLEU-2 (2-gram): {bleu['bleu_2']:.4f}\n")
            f.write(f"BLEU-3 (3-gram): {bleu['bleu_3']:.4f}\n")
            f.write(f"BLEU-4 (4-gram): {bleu['bleu_4']:.4f}\n")
            f.write("\n")

            # ROUGE
            f.write("ROUGE SCORES (Recall-based)\n")
            f.write("-" * 80 + "\n")
            rouge = metrics['rouge']
            f.write(f"ROUGE-1:         {rouge['rouge1']:.4f}\n")
            f.write(f"ROUGE-2:         {rouge['rouge2']:.4f}\n")
            f.write(f"ROUGE-L:         {rouge['rougeL']:.4f}\n")
            f.write("\n")

            # Semantic fidelity
            f.write("SEMANTIC FIDELITY (Cosine Similarity)\n")
            f.write("-" * 80 + "\n")
            fidelity = metrics['semantic_fidelity']
            f.write(f"Mean:            {fidelity['mean_similarity']:.4f}\n")
            f.write(f"Median:          {fidelity['median_similarity']:.4f}\n")
            f.write(f"Std Dev:         {fidelity['std_similarity']:.4f}\n")
            f.write(f"Min:             {fidelity['min_similarity']:.4f}\n")
            f.write(f"Max:             {fidelity['max_similarity']:.4f}\n")
            f.write("\n")

            # Example reconstructions
            if 'reconstructed_texts' in metrics:
                f.write("EXAMPLE RECONSTRUCTIONS (First 5)\n")
                f.write("-" * 80 + "\n")
                for i, text in enumerate(metrics['reconstructed_texts'][:5], 1):
                    display_text = text[:100] + "..." if len(text) > 100 else text
                    f.write(f"[{i}] {display_text}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")

    def evaluate_from_dataset(
        self,
        dataset,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
        save_results: bool = True,
        save_reconstructions: bool = False
    ) -> Dict[str, any]:
        """
        Evaluate using a dataset object.

        Args:
            dataset: Dataset with text samples (MOSIRawVideoDataset, etc.)
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to evaluate (None = all)
            save_results: Whether to save results to disk
            save_reconstructions: Whether to save reconstructed outputs

        Returns:
            Dict with evaluation results
        """
        # Extract text samples from dataset
        test_texts = []
        num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

        if self.verbose:
            print(f"Extracting {num_samples} samples from dataset...")
            from tqdm import tqdm
            sample_iter = tqdm(range(num_samples), desc="Loading samples")
        else:
            sample_iter = range(num_samples)

        for i in sample_iter:
            sample = dataset[i]
            # Assume dataset returns dict with 'text' field
            if isinstance(sample, dict) and 'text' in sample:
                test_texts.append(sample['text'])
            elif isinstance(sample, str):
                test_texts.append(sample)
            else:
                raise ValueError(f"Unsupported sample format: {type(sample)}")

        if self.verbose:
            print(f"[OK] Loaded {len(test_texts)} samples")
            print()

        # Run evaluation
        return self.evaluate(
            test_data=test_texts,
            batch_size=batch_size,
            save_results=save_results,
            save_reconstructions=save_reconstructions
        )

    def compare_checkpoints(
        self,
        checkpoint_paths: List[str],
        test_data: List[str],
        batch_size: int = 32
    ) -> Dict[str, any]:
        """
        Compare multiple decoder checkpoints on the same test data.

        Args:
            checkpoint_paths: List of adapter checkpoint paths
            test_data: Test samples to evaluate on
            batch_size: Batch size for processing

        Returns:
            Dict with comparative results for all checkpoints
        """
        if self.verbose:
            print("=" * 80)
            print("CHECKPOINT COMPARISON")
            print("=" * 80)
            print(f"Comparing {len(checkpoint_paths)} checkpoints")
            print(f"Test samples: {len(test_data)}")
            print()

        results = {}

        for i, checkpoint_path in enumerate(checkpoint_paths, 1):
            if self.verbose:
                print(f"[{i}/{len(checkpoint_paths)}] Evaluating: {checkpoint_path}")

            # Load checkpoint
            self.decoder.adapter.load_state_dict(torch.load(checkpoint_path))
            self.decoder.eval()

            # Run evaluation (don't save individual results)
            checkpoint_results = self.evaluate(
                test_data=test_data,
                batch_size=batch_size,
                save_results=False,
                save_reconstructions=False
            )

            checkpoint_name = os.path.basename(checkpoint_path)
            results[checkpoint_name] = checkpoint_results['metrics']

            if self.verbose:
                print(f"    BLEU: {checkpoint_results['metrics']['bleu']['bleu']:.4f}")
                print(f"    ROUGE-L: {checkpoint_results['metrics']['rouge']['rougeL']:.4f}")
                print(f"    Semantic fidelity: {checkpoint_results['metrics']['semantic_fidelity']['mean_similarity']:.4f}")
                print()

        # Save comparison results
        comparison_path = os.path.join(self.output_dir, f"checkpoint_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        serializable_results = {}
        for checkpoint_name, metrics in results.items():
            serializable_results[checkpoint_name] = {
                'bleu': metrics['bleu'],
                'rouge': metrics['rouge'],
                'semantic_fidelity': metrics['semantic_fidelity'],
                'reconstruction_quality': metrics['reconstruction_quality']
            }

        with open(comparison_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        if self.verbose:
            print(f"[OK] Comparison results saved: {comparison_path}")
            print()

        return results


def load_evaluator_from_config(config_path: str) -> DecoderEvaluator:
    """
    Load evaluator from JSON configuration file.

    Config format:
    {
        "decoder_type": "Vec_to_Text",
        "encoder_type": "Text_to_Vec",
        "output_dir": "results/decoder_evaluation/",
        "device": "cuda",
        "verbose": true
    }

    Args:
        config_path: Path to JSON config file

    Returns:
        Configured DecoderEvaluator instance
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Import decoder and encoder dynamically
    decoder_type = config['decoder_type']
    encoder_type = config['encoder_type']

    # Map type names to module paths
    decoder_map = {
        'Vec_to_Text': 'src.Decoders.Vec_to_Text',
        'Vec_to_Audio': 'src.Decoders.Vec_to_Audio',
        'Vec_to_Image': 'src.Decoders.Vec_to_Image'
    }

    encoder_map = {
        'Text_to_Vec': 'src.Encoders.Text_to_Vec',
        'Audio_to_Vec': 'src.Encoders.Audio_to_Vec',
        'Image_to_Vec': 'src.Encoders.Image_to_Vec'
    }

    # Dynamic import
    from importlib import import_module

    decoder_module_path = decoder_map.get(decoder_type)
    encoder_module_path = encoder_map.get(encoder_type)

    if not decoder_module_path or not encoder_module_path:
        raise ValueError(f"Unknown decoder/encoder type: {decoder_type}, {encoder_type}")

    decoder_module = import_module(decoder_module_path)
    encoder_module = import_module(encoder_module_path)

    decoder_class = getattr(decoder_module, decoder_type)
    encoder_class = getattr(encoder_module, encoder_type)

    # Create evaluator
    return DecoderEvaluator(
        decoder=decoder_class(),
        encoder=encoder_class(),
        output_dir=config.get('output_dir', 'results/decoder_evaluation/'),
        device=config.get('device'),
        verbose=config.get('verbose', True)
    )


if __name__ == "__main__":
    # Example usage
    print("Decoder Evaluator - Example Usage")
    print()
    print("This module provides a comprehensive evaluation pipeline for decoder modules.")
    print()
    print("Basic usage:")
    print("  from src.Evaluation.decoder_evaluator import DecoderEvaluator")
    print("  from src.Encoders import Text_to_Vec")
    print("  from src.Decoders import Vec_to_Text")
    print()
    print("  evaluator = DecoderEvaluator(")
    print("      decoder=Vec_to_Text(),")
    print("      encoder=Text_to_Vec(),")
    print("      output_dir='results/decoder_evaluation/'")
    print("  )")
    print()
    print("  results = evaluator.evaluate(")
    print("      test_data=['example text 1', 'example text 2', ...],")
    print("      batch_size=32")
    print("  )")
    print()
    print("For standalone evaluation, use: python scripts/evaluate_decoder.py")
    print()
