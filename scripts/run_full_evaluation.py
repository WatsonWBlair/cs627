#!/usr/bin/env python
"""
Comprehensive Evaluation Pipeline

This script runs complete evaluation of trained encoders and decoders with:
1. Cross-modal retrieval metrics
2. Semantic clustering analysis
3. Decoder reconstruction quality
4. Benchmark evaluation (MTEB, GLUE if available)
5. Automatic figure generation and report creation

Usage:
    # Basic evaluation
    python scripts/run_full_evaluation.py
    
    # Evaluation with custom parameters
    python scripts/run_full_evaluation.py --encoders text audio image --max-samples 1000
    
    # Include decoder evaluation
    python scripts/run_full_evaluation.py --evaluate-decoders
    
    # Generate publication figures
    python scripts/run_full_evaluation.py --publication-quality --output-dir results/paper_figures/
"""

import sys
import os
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import evaluation modules
from src.Evaluation import (
    compute_cross_modal_metrics,
    print_metrics_summary,
    CrossModalEvaluator
)
from src.Evaluation.visualization import (
    plot_retrieval_confusion_matrix,
    plot_semantic_clustering,
    plot_similarity_heatmap,
    plot_performance_comparison,
    plot_decoder_reconstruction,
    create_evaluation_dashboard
)
from src.Evaluation.decoder_metrics import (
    evaluate_text_decoder,
    print_decoder_metrics_summary
)

# Import encoders and decoders
from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from src.Decoders import Vec_to_Text

# Import datasets
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
from src.Training.Data_Wrangling.benchmark_dataset import BenchmarkDataset, collate_fn_benchmark
from src.Training.train_encoders import collate_fn_raw_video


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation of encoders and decoders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Encoder selection
    parser.add_argument(
        '--encoders',
        nargs='+',
        default=['text', 'audio', 'image'],
        choices=['text', 'audio', 'image', 'tone'],
        help='Which encoders to evaluate'
    )
    
    # Decoder evaluation
    parser.add_argument(
        '--evaluate-decoders',
        action='store_true',
        help='Also evaluate decoder modules'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset',
        type=str,
        default='mosi',
        choices=['mosi', 'benchmark', 'both'],
        help='Which dataset to use for evaluation'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'valid', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to evaluate (None = all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/full_evaluation/',
        help='Directory to save results'
    )
    parser.add_argument(
        '--publication-quality',
        action='store_true',
        help='Generate publication-quality figures (higher DPI, better layout)'
    )
    parser.add_argument(
        '--save-embeddings',
        action='store_true',
        help='Save computed embeddings to disk'
    )
    
    # Visualization options
    parser.add_argument(
        '--clustering-method',
        type=str,
        default='tsne',
        choices=['tsne', 'umap', 'pca'],
        help='Dimensionality reduction method for clustering visualization'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization generation (faster)'
    )
    
    # Data paths
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
        help='Path to audio files'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/cmumosi/frames/',
        help='Path to video frames'
    )
    
    return parser.parse_args()


def load_encoders(encoder_names: List[str], device: str = 'cuda') -> Dict:
    """Load trained encoder modules."""
    print("\n" + "="*80)
    print("LOADING ENCODERS")
    print("="*80)
    
    encoders = {}
    
    for name in encoder_names:
        try:
            if name == 'text':
                print("Loading Text_to_Vec encoder...")
                encoders['text'] = Text_to_Vec().to(device)
                print("  ✓ Text encoder loaded")
                
            elif name == 'audio':
                print("Loading Audio_to_Vec encoder...")
                encoders['audio'] = Audio_to_Vec().to(device)
                print("  ✓ Audio encoder loaded")
                
            elif name == 'image':
                print("Loading Image_to_Vec encoder...")
                encoders['image'] = Image_to_Vec().to(device)
                print("  ✓ Image encoder loaded")
                
            elif name == 'tone':
                print("Loading Tone_to_Vec encoder...")
                from src.Encoders.audio.tone_to_vec import Tone_to_Vec
                encoders['tone'] = Tone_to_Vec().to(device)
                print("  ✓ Tone encoder loaded")
                
        except Exception as e:
            print(f"  ✗ Failed to load {name} encoder: {e}")
    
    print(f"\nSuccessfully loaded {len(encoders)}/{len(encoder_names)} encoders")
    
    # Set to eval mode
    for encoder in encoders.values():
        encoder.eval()
    
    return encoders


def load_decoders(decoder_names: List[str], device: str = 'cuda') -> Dict:
    """Load trained decoder modules."""
    print("\n" + "="*80)
    print("LOADING DECODERS")
    print("="*80)
    
    decoders = {}
    
    for name in decoder_names:
        try:
            if name == 'text':
                print("Loading Vec_to_Text decoder...")
                decoders['text'] = Vec_to_Text().to(device)
                print("  ✓ Text decoder loaded")
                
            # Add other decoders as they become available
            
        except Exception as e:
            print(f"  ✗ Failed to load {name} decoder: {e}")
    
    print(f"\nSuccessfully loaded {len(decoders)}/{len(decoder_names)} decoders")
    
    # Set to eval mode
    for decoder in decoders.values():
        decoder.eval()
    
    return decoders


def load_dataset(args):
    """Load evaluation dataset."""
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    if args.dataset == 'mosi':
        print("Loading MOSI dataset...")
        dataset = MOSIRawVideoDataset(
            split=args.split,
            mosi_data_path=args.mosi_data_path,
            audio_dir=args.audio_dir,
            video_dir=args.video_dir
        )
        collate_fn = collate_fn_raw_video
        
    elif args.dataset == 'benchmark':
        print("Loading benchmark dataset (MTEB + GLUE)...")
        dataset = BenchmarkDataset(
            data_sources=['mteb', 'glue'],
            split=args.split,
            modalities=['text']
        )
        collate_fn = collate_fn_benchmark
        
    else:  # both
        print("Loading combined dataset...")
        dataset = BenchmarkDataset(
            data_sources=['mosi', 'mteb', 'glue'],
            split=args.split,
            modalities=['text', 'audio', 'video'],
            mosi_data_path=args.mosi_data_path,
            mosi_audio_dir=args.audio_dir,
            mosi_video_dir=args.video_dir
        )
        collate_fn = collate_fn_benchmark
    
    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Split: {args.split}")
    
    return dataset, collate_fn


def evaluate_encoders(encoders, dataset, collate_fn, args, output_dir):
    """Run encoder evaluation."""
    print("\n" + "="*80)
    print("ENCODER EVALUATION")
    print("="*80)
    
    # Create evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = CrossModalEvaluator(encoders, device=device)
    
    # Set collate function
    dataset.collate_fn = collate_fn
    
    # Run evaluation
    print("\nRunning cross-modal evaluation...")
    metrics = evaluator.evaluate(
        dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        k_values=[1, 5, 10],
        save_results=None  # We'll save manually
    )
    
    # Save raw metrics
    metrics_file = output_dir / 'encoder_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_metrics = json.loads(json.dumps(metrics, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))
        json.dump(json_metrics, f, indent=2)
    print(f"Saved encoder metrics to {metrics_file}")
    
    # Save embeddings if requested
    if args.save_embeddings:
        print("\nSaving embeddings...")
        embeddings = evaluator.encode_dataset(
            dataset,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        
        embeddings_file = output_dir / 'embeddings.pt'
        torch.save(embeddings, embeddings_file)
        print(f"Saved embeddings to {embeddings_file}")
        
        return metrics, embeddings
    
    return metrics, None


def evaluate_decoders(decoders, encoders, dataset, args, output_dir):
    """Run decoder evaluation."""
    print("\n" + "="*80)
    print("DECODER EVALUATION")
    print("="*80)
    
    results = {}
    
    # Evaluate text decoder if available
    if 'text' in decoders and 'text' in encoders:
        print("\nEvaluating Text decoder...")
        
        # Get test texts
        test_texts = []
        for i in range(min(100, len(dataset))):  # Use subset for faster eval
            sample = dataset[i]
            if 'text' in sample:
                test_texts.append(sample['text'])
            elif 'anchor_text' in sample:
                test_texts.append(sample['anchor_text'])
        
        if test_texts:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            metrics = evaluate_text_decoder(
                decoders['text'],
                encoders['text'],
                test_texts,
                device=device,
                batch_size=args.batch_size,
                verbose=True
            )
            
            results['text'] = metrics
            
            # Save metrics
            decoder_metrics_file = output_dir / 'decoder_metrics_text.json'
            with open(decoder_metrics_file, 'w') as f:
                # Remove reconstructed texts for JSON serialization
                json_metrics = {k: v for k, v in metrics.items() if k != 'reconstructed_texts'}
                json.dump(json_metrics, f, indent=2)
            print(f"Saved text decoder metrics to {decoder_metrics_file}")
    
    # Add evaluation for other decoders as they become available
    
    return results


def generate_visualizations(metrics, embeddings, args, output_dir):
    """Generate all visualization figures."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set DPI based on quality setting
    dpi = 300 if args.publication_quality else 150
    
    # 1. Retrieval confusion matrix
    if 'retrieval' in metrics:
        print("Creating retrieval confusion matrix...")
        fig = plot_retrieval_confusion_matrix(
            metrics['retrieval'],
            save_path=figures_dir / 'retrieval_confusion.png'
        )
        plt.close(fig)
    
    # 2. Similarity heatmap
    if 'similarity_matrix' in metrics:
        print("Creating similarity heatmap...")
        fig = plot_similarity_heatmap(
            np.array(metrics['similarity_matrix']['matrix']),
            metrics['similarity_matrix']['modality_names'],
            save_path=figures_dir / 'similarity_heatmap.png'
        )
        plt.close(fig)
    
    # 3. Semantic clustering (if embeddings available)
    if embeddings is not None:
        print(f"Creating semantic clustering ({args.clustering_method})...")
        fig = plot_semantic_clustering(
            embeddings,
            method=args.clustering_method,
            save_path=figures_dir / f'clustering_{args.clustering_method}.png',
            max_samples=500  # Limit for performance
        )
        plt.close(fig)
    
    # 4. Performance comparison
    if 'alignment_quality' in metrics:
        print("Creating performance comparison...")
        # Format data for comparison plot
        perf_data = {}
        for modality, quality in metrics['alignment_quality'].items():
            perf_data[modality] = {
                'alignment_score': quality['alignment_score'],
                'mean_distance': 1.0 - quality['mean_distance']  # Convert to similarity
            }
        
        fig = plot_performance_comparison(
            perf_data,
            save_path=figures_dir / 'performance_comparison.png'
        )
        plt.close(fig)
    
    print(f"Visualizations saved to {figures_dir}")
    
    # Create HTML dashboard
    dashboard_path = create_evaluation_dashboard(
        metrics,
        output_dir=output_dir,
        title=f"Evaluation Report - {args.dataset.upper()} {args.split}"
    )
    
    return dashboard_path


def generate_summary_report(encoder_metrics, decoder_metrics, args, output_dir):
    """Generate comprehensive summary report."""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    report_lines = [
        "# COMPREHENSIVE EVALUATION REPORT",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {args.dataset} ({args.split} split)",
        f"Samples evaluated: {args.max_samples or 'all'}",
        "",
        "## ENCODER EVALUATION",
        "-" * 40
    ]
    
    # Add encoder metrics
    if 'retrieval' in encoder_metrics:
        report_lines.append("\n### Cross-Modal Retrieval (Average)")
        
        all_r1 = [m.get('R@1', 0) for m in encoder_metrics['retrieval'].values()]
        all_r5 = [m.get('R@5', 0) for m in encoder_metrics['retrieval'].values()]
        all_r10 = [m.get('R@10', 0) for m in encoder_metrics['retrieval'].values()]
        
        if all_r1:
            report_lines.append(f"  R@1:  {np.mean(all_r1):.3f}")
            report_lines.append(f"  R@5:  {np.mean(all_r5):.3f}")
            report_lines.append(f"  R@10: {np.mean(all_r10):.3f}")
        
        # Add per-pair results
        report_lines.append("\n### Retrieval by Pair")
        for pair, metrics in sorted(encoder_metrics['retrieval'].items()):
            report_lines.append(f"  {pair}: R@1={metrics['R@1']:.3f}, R@5={metrics['R@5']:.3f}")
    
    if 'alignment_quality' in encoder_metrics:
        report_lines.append("\n### Alignment Quality")
        for modality, quality in encoder_metrics['alignment_quality'].items():
            report_lines.append(f"  {modality}: {quality['alignment_score']:.3f}")
    
    # Add decoder metrics
    if decoder_metrics:
        report_lines.append("\n## DECODER EVALUATION")
        report_lines.append("-" * 40)
        
        for modality, metrics in decoder_metrics.items():
            report_lines.append(f"\n### {modality.upper()} Decoder")
            if 'bleu' in metrics:
                report_lines.append(f"  BLEU: {metrics['bleu']['bleu']:.3f}")
            if 'rouge' in metrics:
                report_lines.append(f"  ROUGE-L: {metrics['rouge']['rougeL']:.3f}")
            if 'semantic_fidelity' in metrics:
                report_lines.append(f"  Semantic Fidelity: {metrics['semantic_fidelity']['mean_similarity']:.3f}")
    
    # Add configuration
    report_lines.extend([
        "\n## CONFIGURATION",
        "-" * 40,
        f"  Encoders: {args.encoders}",
        f"  Batch size: {args.batch_size}",
        f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}",
        f"  Output directory: {output_dir}",
    ])
    
    # Save report
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to {report_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for line in report_lines[6:20]:  # Print key metrics
        print(line)


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"eval_{args.dataset}_{args.split}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Output directory: {output_dir}")
    
    # Save configuration
    config = vars(args)
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load models
    encoders = load_encoders(args.encoders, device)
    
    decoders = {}
    if args.evaluate_decoders:
        # Currently only text decoder is available
        decoder_names = ['text'] if 'text' in args.encoders else []
        decoders = load_decoders(decoder_names, device)
    
    # Load dataset
    dataset, collate_fn = load_dataset(args)
    
    # Run encoder evaluation
    encoder_metrics, embeddings = evaluate_encoders(
        encoders, dataset, collate_fn, args, output_dir
    )
    
    # Run decoder evaluation
    decoder_metrics = {}
    if args.evaluate_decoders and decoders:
        decoder_metrics = evaluate_decoders(
            decoders, encoders, dataset, args, output_dir
        )
    
    # Generate visualizations
    if not args.skip_visualization:
        import matplotlib.pyplot as plt
        dashboard_path = generate_visualizations(
            encoder_metrics, embeddings, args, output_dir
        )
        print(f"\nDashboard created: {dashboard_path}")
    
    # Generate summary report
    generate_summary_report(
        encoder_metrics, decoder_metrics, args, output_dir
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the summary report: summary_report.txt")
    if not args.skip_visualization:
        print("2. Open the dashboard: dashboard.html")
        print("3. Check figures in: figures/")
    print("4. Analyze detailed metrics: encoder_metrics.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())