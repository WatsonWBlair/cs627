"""
Cross-Modal Evaluator

Evaluates encoder alignment by encoding a dataset with all modality encoders
and computing cross-modal retrieval metrics.

Usage:
    evaluator = CrossModalEvaluator(encoders, dataset)
    results = evaluator.evaluate(split='test', batch_size=32)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from tqdm import tqdm

from ..encoder_metrics import compute_cross_modal_metrics, print_metrics_summary


class CrossModalEvaluator:
    """
    Evaluates cross-modal encoder alignment on a dataset.

    Encodes all samples with all available encoders, then computes
    cross-modal retrieval metrics (Recall@K), similarity matrices,
    alignment quality, and modality gaps.
    """

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator with trained encoders.

        Args:
            encoders: Dict mapping encoder_name to encoder module
                     e.g., {"text": Text_to_Vec(), "audio": Audio_to_Vec()}
            device: Device to run evaluation on
        """
        self.encoders = encoders
        self.device = device

        # Move encoders to device and set to eval mode
        for name, encoder in self.encoders.items():
            encoder.to(device)
            encoder.eval()

        print(f"[CrossModalEvaluator] Initialized with {len(encoders)} encoders:")
        for name in encoders.keys():
            print(f"  - {name}")

    @torch.no_grad()
    def encode_dataset(
        self,
        dataset,
        batch_size: int = 32,
        max_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode entire dataset with all available encoders.

        Args:
            dataset: PyTorch Dataset that returns dict with modality keys
            batch_size: Batch size for encoding
            max_samples: Optional limit on number of samples to encode

        Returns:
            Dict mapping encoder_name to (N, D) tensor of embeddings
        """
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=getattr(dataset, 'collate_fn', None)
        )

        # Initialize storage for embeddings
        embeddings_dict = {name: [] for name in self.encoders.keys()}

        # Track which modalities are actually available in dataset
        available_modalities = set()

        print(f"[CrossModalEvaluator] Encoding dataset...")
        num_samples = 0

        for batch in tqdm(dataloader, desc="Encoding batches"):
            # Check available modalities from first batch
            if not available_modalities:
                for encoder_name, encoder in self.encoders.items():
                    # Determine which data key this encoder needs
                    # Convention: text_base uses 'text', audio_* uses 'audio', image_* uses 'video'
                    if 'text' in encoder_name:
                        data_key = 'text'
                    elif 'audio' in encoder_name:
                        data_key = 'audio'
                    elif 'image' in encoder_name or 'visual' in encoder_name:
                        data_key = 'video'
                    else:
                        # Try to infer from encoder name
                        data_key = encoder_name.split('_')[0]

                    if data_key in batch and batch[data_key] is not None:
                        available_modalities.add(encoder_name)

            # Encode with each available encoder
            for encoder_name in available_modalities:
                encoder = self.encoders[encoder_name]

                # Get data for this encoder
                if 'text' in encoder_name:
                    data = batch['text']
                elif 'audio' in encoder_name:
                    data = batch['audio']
                elif 'image' in encoder_name or 'visual' in encoder_name:
                    data = batch['video']
                else:
                    continue

                # Handle different input types
                if isinstance(data, list):
                    # Text data (list of strings)
                    embeddings = encoder(data)
                elif isinstance(data, str):
                    # Single text string
                    embeddings = encoder([data])
                elif torch.is_tensor(data):
                    # Tensor data (audio/image)
                    embeddings = encoder(data.to(self.device))
                else:
                    print(f"[WARNING] Unsupported data type for {encoder_name}: {type(data)}")
                    continue

                embeddings_dict[encoder_name].append(embeddings.cpu())

            num_samples += batch['text'].__len__() if isinstance(batch['text'], list) else 1

            # Check sample limit
            if max_samples and num_samples >= max_samples:
                break

        # Concatenate all embeddings
        final_embeddings = {}
        for encoder_name in available_modalities:
            if embeddings_dict[encoder_name]:
                final_embeddings[encoder_name] = torch.cat(embeddings_dict[encoder_name], dim=0)
                print(f"  {encoder_name}: {final_embeddings[encoder_name].shape}")

        print(f"[CrossModalEvaluator] Encoded {num_samples} samples")

        return final_embeddings

    def evaluate(
        self,
        dataset,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
        k_values: List[int] = [1, 5, 10],
        save_results: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Run full cross-modal evaluation on dataset.

        Args:
            dataset: PyTorch Dataset to evaluate on
            batch_size: Batch size for encoding
            max_samples: Optional limit on samples
            k_values: K values for Recall@K
            save_results: Optional path to save results JSON

        Returns:
            Dict with all evaluation metrics
        """
        print("=" * 80)
        print("CROSS-MODAL ENCODER EVALUATION")
        print("=" * 80)
        print()

        # Step 1: Encode dataset
        embeddings = self.encode_dataset(dataset, batch_size, max_samples)

        if not embeddings:
            raise ValueError("No embeddings were generated! Check dataset and encoders.")

        # Step 2: Compute metrics
        print()
        print("[CrossModalEvaluator] Computing cross-modal metrics...")
        metrics = compute_cross_modal_metrics(embeddings, k_values=k_values)

        # Step 3: Print summary
        print()
        print_metrics_summary(metrics)

        # Step 4: Save results if requested
        if save_results:
            save_path = Path(save_results)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"\n[CrossModalEvaluator] Results saved to: {save_path}")

        return metrics

    def evaluate_pairs(
        self,
        dataset,
        modality_pairs: List[Tuple[str, str]],
        batch_size: int = 32,
        max_samples: Optional[int] = None
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Evaluate specific modality pairs only.

        Useful for focused evaluation when you don't need all pairwise metrics.

        Args:
            dataset: PyTorch Dataset
            modality_pairs: List of (query_modality, target_modality) tuples
            batch_size: Batch size for encoding
            max_samples: Optional sample limit

        Returns:
            Dict mapping (query, target) pair to Recall@K metrics
        """
        # Encode dataset
        embeddings = self.encode_dataset(dataset, batch_size, max_samples)

        # Compute metrics for each pair
        from ..encoder_metrics import compute_recall_at_k

        results = {}
        for query_mod, target_mod in modality_pairs:
            if query_mod not in embeddings or target_mod not in embeddings:
                print(f"[WARNING] Skipping pair ({query_mod}, {target_mod}) - embeddings not found")
                continue

            recall = compute_recall_at_k(
                embeddings[query_mod],
                embeddings[target_mod],
                k_values=[1, 5, 10]
            )
            results[(query_mod, target_mod)] = recall

            print(f"{query_mod} → {target_mod}:")
            print(f"  R@1: {recall['R@1']:.3f}  R@5: {recall['R@5']:.3f}  R@10: {recall['R@10']:.3f}")

        return results


if __name__ == "__main__":
    """
    Example usage with trained encoders and MOSI dataset.
    """
    import sys
    sys.path.insert(0, 'src')

    from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
    from Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset, collate_fn_raw_video

    print("Loading encoders...")
    encoders = {
        "text_base": Text_to_Vec(),
        "audio_waveform": Audio_to_Vec(),
        "image_base": Image_to_Vec()
    }

    print("Loading MOSI test dataset...")
    dataset = MOSIRawVideoDataset(
        split='test',
        mosi_data_path='data/cmumosi/mosi/',
        audio_dir='data/cmumosi/audio/',
        video_dir='data/cmumosi/frames/'
    )
    dataset.collate_fn = collate_fn_raw_video

    print(f"Dataset size: {len(dataset)} samples")
    print()

    # Create evaluator
    evaluator = CrossModalEvaluator(encoders)

    # Run evaluation (limit to 50 samples for quick test)
    results = evaluator.evaluate(
        dataset,
        batch_size=16,
        max_samples=50,
        save_results='results/cross_modal_eval.json'
    )

    print("\nEvaluation complete!")
