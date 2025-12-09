"""
Unified Triplet Dataset for Contrastive Learning

Provides a unified interface for loading triplet data from all wrangled datasets.
Supports on-the-fly combination and PyTorch DataLoader integration.

Usage:
    from unified_triplet_dataset import UnifiedTripletDataset

    # Load from multiple sources
    dataset = UnifiedTripletDataset.from_directories([
        'data/glue/',
        'data/mteb/',
        'data/clinc/'
    ])

    # Use with PyTorch
    dataloader = dataset.to_dataloader(batch_size=32)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Union
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


class TripletDataset(Dataset):
    """
    PyTorch Dataset for triplet data.

    Loads pre-generated triplets from JSON files.
    """

    def __init__(self, triplets: List[Dict[str, str]]):
        """
        Initialize triplet dataset.

        Args:
            triplets: List of triplet dictionaries with keys 'anchor', 'positive', 'negative'
        """
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.triplets[idx]

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'TripletDataset':
        """
        Load triplet dataset from JSON file.

        Args:
            path: Path to JSON file with triplets

        Returns:
            TripletDataset instance
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        triplets = data.get('triplets', data)
        if isinstance(triplets, dict):
            triplets = triplets.get('triplets', [])

        return cls(triplets)


class UnifiedTripletDataset(Dataset):
    """
    Unified dataset combining triplets from multiple sources.

    Provides a single interface for loading and accessing triplet data
    from all wrangled datasets.
    """

    def __init__(
        self,
        triplets: List[Dict[str, str]],
        source_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize unified dataset.

        Args:
            triplets: Combined list of triplet dictionaries
            source_weights: Optional weights for balancing sources
        """
        self.triplets = triplets
        self.source_weights = source_weights or {}

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.triplets[idx]

    @classmethod
    def from_directories(
        cls,
        directories: List[Union[str, Path]],
        max_per_source: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> 'UnifiedTripletDataset':
        """
        Load and combine triplets from multiple directories.

        Args:
            directories: List of directories containing triplet JSON files
            max_per_source: Maximum samples per source (for balancing)
            shuffle: Whether to shuffle the combined data
            seed: Random seed for reproducibility

        Returns:
            UnifiedTripletDataset instance
        """
        random.seed(seed)
        all_triplets = []
        source_counts = defaultdict(int)

        for dir_path in directories:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                print(f"Warning: Directory not found: {dir_path}")
                continue

            # Find triplet JSON files
            for json_file in dir_path.glob('*_triplets.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    triplets = data.get('triplets', [])
                    source = data.get('metadata', {}).get('source', json_file.stem)

                    # Apply max_per_source limit
                    if max_per_source and len(triplets) > max_per_source:
                        triplets = random.sample(triplets, max_per_source)

                    # Add source metadata to each triplet
                    for triplet in triplets:
                        triplet['_source'] = source

                    all_triplets.extend(triplets)
                    source_counts[source] = len(triplets)
                    print(f"Loaded {len(triplets)} triplets from {json_file.name}")

                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

        if shuffle:
            random.shuffle(all_triplets)

        print(f"\nTotal triplets: {len(all_triplets)}")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")

        return cls(all_triplets, source_weights=dict(source_counts))

    @classmethod
    def from_files(
        cls,
        files: List[Union[str, Path]],
        max_per_source: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> 'UnifiedTripletDataset':
        """
        Load and combine triplets from specific files.

        Args:
            files: List of JSON files to load
            max_per_source: Maximum samples per source
            shuffle: Whether to shuffle combined data
            seed: Random seed

        Returns:
            UnifiedTripletDataset instance
        """
        random.seed(seed)
        all_triplets = []
        source_counts = defaultdict(int)

        for file_path in files:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                triplets = data.get('triplets', [])
                source = data.get('metadata', {}).get('source', file_path.stem)

                if max_per_source and len(triplets) > max_per_source:
                    triplets = random.sample(triplets, max_per_source)

                for triplet in triplets:
                    triplet['_source'] = source

                all_triplets.extend(triplets)
                source_counts[source] = len(triplets)
                print(f"Loaded {len(triplets)} triplets from {file_path.name}")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if shuffle:
            random.shuffle(all_triplets)

        return cls(all_triplets, source_weights=dict(source_counts))

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple['UnifiedTripletDataset', 'UnifiedTripletDataset', 'UnifiedTripletDataset']:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed

        Returns:
            Tuple of (train, val, test) datasets
        """
        random.seed(seed)
        shuffled = self.triplets.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = UnifiedTripletDataset(shuffled[:train_end], self.source_weights)
        val = UnifiedTripletDataset(shuffled[train_end:val_end], self.source_weights)
        test = UnifiedTripletDataset(shuffled[val_end:], self.source_weights)

        return train, val, test

    def to_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        Create PyTorch DataLoader.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs
        )

    @staticmethod
    def _collate_fn(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Collate batch of triplets.

        Args:
            batch: List of triplet dictionaries

        Returns:
            Dictionary with lists for each key
        """
        return {
            'anchor': [item['anchor'] for item in batch],
            'positive': [item['positive'] for item in batch],
            'negative': [item['negative'] for item in batch]
        }

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_triplets': len(self.triplets),
            'sources': self.source_weights,
            'avg_anchor_length': 0,
            'avg_positive_length': 0,
            'avg_negative_length': 0
        }

        if self.triplets:
            stats['avg_anchor_length'] = sum(
                len(t['anchor'].split()) for t in self.triplets
            ) / len(self.triplets)
            stats['avg_positive_length'] = sum(
                len(t['positive'].split()) for t in self.triplets
            ) / len(self.triplets)
            stats['avg_negative_length'] = sum(
                len(t['negative'].split()) for t in self.triplets
            ) / len(self.triplets)

        return stats

    def save(self, path: Union[str, Path]) -> None:
        """
        Save combined dataset to JSON.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Remove internal metadata before saving
        clean_triplets = [
            {k: v for k, v in t.items() if not k.startswith('_')}
            for t in self.triplets
        ]

        output = {
            'triplets': clean_triplets,
            'metadata': {
                'source': 'unified',
                'total_samples': len(clean_triplets),
                'sources': self.source_weights
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(clean_triplets)} triplets to {path}")


class StreamingTripletDataset(IterableDataset):
    """
    Streaming triplet dataset for memory-efficient training.

    Iterates over triplet files without loading all data into memory.
    """

    def __init__(
        self,
        files: List[Union[str, Path]],
        shuffle_files: bool = True,
        seed: int = 42
    ):
        """
        Initialize streaming dataset.

        Args:
            files: List of triplet JSON files
            shuffle_files: Whether to shuffle file order
            seed: Random seed
        """
        self.files = [Path(f) for f in files]
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Iterate over all triplets."""
        files = self.files.copy()
        if self.shuffle_files:
            random.seed(self.seed)
            random.shuffle(files)

        for file_path in files:
            if not file_path.exists():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                triplets = data.get('triplets', [])
                for triplet in triplets:
                    yield triplet

            except Exception as e:
                print(f"Error reading {file_path}: {e}")


def main():
    """Demo usage of unified triplet dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and combine triplet datasets"
    )
    parser.add_argument(
        '--data-dirs',
        nargs='+',
        default=['data/glue/', 'data/mteb/', 'data/clinc/', 'data/meld/'],
        help='Directories containing triplet data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/unified_triplets.json',
        help='Output file for combined dataset'
    )
    parser.add_argument(
        '--max-per-source',
        type=int,
        default=None,
        help='Maximum samples per source'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print dataset statistics'
    )

    args = parser.parse_args()

    # Load from directories
    dataset = UnifiedTripletDataset.from_directories(
        args.data_dirs,
        max_per_source=args.max_per_source
    )

    if args.stats:
        stats = dataset.get_statistics()
        print("\nDataset Statistics:")
        print(f"  Total triplets: {stats['total_triplets']}")
        print(f"  Avg anchor length: {stats['avg_anchor_length']:.1f} words")
        print(f"  Avg positive length: {stats['avg_positive_length']:.1f} words")
        print(f"  Avg negative length: {stats['avg_negative_length']:.1f} words")

    if args.output:
        dataset.save(args.output)


if __name__ == "__main__":
    main()
