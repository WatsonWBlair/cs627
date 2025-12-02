"""
CMU-MOSI Dataset Loader for Cross-Modal Contrastive Learning

This module provides utilities for loading and preprocessing the CMU-MOSI dataset
for training cross-modal encoders using momentum contrastive learning.

Dataset: CMU Multimodal Opinion Sentiment Intensity (MOSI)
- 2,199 opinion video segments from YouTube
- Modalities: Text transcripts, Audio, Video
- Labels: Sentiment intensity scores

Usage:
    from Training.Data_Wrangling.mosi_dataset import MOSIDataset, download_mosi

    # Download dataset (first time only)
    download_mosi('data/cmumosi/')

    # Create dataset
    train_dataset = MOSIDataset(split='train', data_path='data/cmumosi/')

    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pickle


def mosi_collate_fn(batch):
    """
    Custom collate function for variable-length MOSI sequences.

    Since sequences have different lengths, we can't stack them directly.
    Instead, return lists of tensors that the encoder will handle individually.
    """
    # Extract each modality into separate lists
    text_features = [torch.tensor(item['text'], dtype=torch.float32) for item in batch]
    audio_features = [torch.tensor(item['audio'], dtype=torch.float32) for item in batch]
    video_features = [torch.tensor(item['video'], dtype=torch.float32) for item in batch]
    segment_ids = [item['segment_id'] for item in batch]

    # Labels can be stacked (they're scalars)
    if 'label' in batch[0]:
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
        return {
            'text': text_features,
            'audio': audio_features,
            'video': video_features,
            'segment_id': segment_ids,
            'label': labels
        }
    else:
        return {
            'text': text_features,
            'audio': audio_features,
            'video': video_features,
            'segment_id': segment_ids
        }


def download_mosi(data_path: str = 'data/cmumosi/'):
    """
    Download CMU-MOSI dataset using CMU-MultimodalSDK

    Args:
        data_path: Path to store downloaded data

    Returns:
        Dictionary containing dataset metadata
    """
    try:
        from mmsdk import mmdatasdk
    except ImportError:
        raise ImportError(
            "CMU-MultimodalSDK not installed. Please install it:\n"
            "git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git\n"
            "cd CMU-MultimodalSDK && pip install ."
        )

    # Use Unix-style relative path for SDK compatibility
    # The SDK expects relative paths with forward slashes (works on Windows and Unix)
    data_path = data_path.replace('\\', '/')
    if not data_path.endswith('/'):
        data_path = data_path + '/'

    os.makedirs(data_path, exist_ok=True)

    print(f"Downloading CMU-MOSI dataset to {data_path}...")
    
    # Download raw data (transcripts, phonemes, etc.)
    print("Downloading raw data (text transcripts)...")
    raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, data_path)

    # Download high-level features
    print("Downloading high-level features...")
    highlevel_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, data_path)

    # Download labels
    print("Downloading labels...")
    highlevel_data.add_computational_sequences(mmdatasdk.cmu_mosi.labels, data_path)

    # Align all data to opinion segment labels
    print("Aligning data to opinion segments...")
    highlevel_data.align('Opinion Segment Labels')

    print(f"[OK] Download complete! Data saved to {data_path}")
    print(f"  Available sequences: {list(highlevel_data.computational_sequences.keys())}")

    return {
        'raw': raw_data,
        'highlevel': highlevel_data,
        'num_segments': len(list(highlevel_data.computational_sequences.values())[0].data.keys())
    }


class MOSIDataset(Dataset):
    """
    PyTorch Dataset for CMU-MOSI multimodal data

    Returns aligned triplets of (text, audio, video) for cross-modal contrastive learning.
    """

    def __init__(
        self,
        data_path: str = 'data/cmumosi/',
        split: str = 'train',
        return_labels: bool = False,
        max_text_length: int = 512
    ):
        """
        Args:
            data_path: Path to CMU-MOSI data directory
            split: Dataset split ('train', 'valid', or 'test')
            return_labels: If True, return sentiment labels
            max_text_length: Maximum length for text sequences
        """
        super().__init__()

        # Normalize to Unix-style path
        self.data_path = data_path.replace('\\', '/')
        if not self.data_path.endswith('/'):
            self.data_path = self.data_path + '/'
        self.split = split
        self.return_labels = return_labels
        self.max_text_length = max_text_length

        # Load preprocessed data if available
        preprocessed_path = os.path.join(self.data_path, f'preprocessed_{split}.pkl')
        if os.path.exists(preprocessed_path):
            print(f"Loading preprocessed {split} data from {preprocessed_path}")
            with open(preprocessed_path, 'rb') as f:
                data = pickle.load(f)
                self.samples = data['samples']
                self.segment_ids = data['segment_ids']
        else:
            # Load and preprocess from scratch
            print(f"Preprocessed data not found. Loading raw data...")
            print(f"NOTE: You may need to run preprocess_mosi() first to prepare the dataset.")
            self.samples = []
            self.segment_ids = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing aligned modalities for one segment

        Returns:
            dict with keys:
                - 'text': Text transcript (str)
                - 'audio': Audio features or waveform (tensor)
                - 'video': Video frame features or image (tensor)
                - 'segment_id': Unique segment identifier (str)
                - 'label': Sentiment intensity (float, if return_labels=True)
        """
        if len(self.samples) == 0:
            raise RuntimeError(
                "No samples loaded. Please run download_mosi() and preprocess_mosi() first."
            )

        sample = self.samples[idx]

        output = {
            'text': sample['text'],
            'audio': sample['audio'],
            'video': sample['video'],
            'segment_id': self.segment_ids[idx]
        }

        if self.return_labels:
            output['label'] = sample['label']

        return output


def preprocess_mosi(
    data_path: str = 'data/cmumosi/',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    Preprocess CMU-MOSI dataset and create train/val/test splits

    This function:
    1. Loads raw MOSI data
    2. Extracts text transcripts, audio features, video features
    3. Splits into train/val/test sets
    4. Saves preprocessed data for faster loading

    Args:
        data_path: Path to CMU-MOSI data directory
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
    """
    try:
        from mmsdk import mmdatasdk
    except ImportError:
        raise ImportError(
            "CMU-MultimodalSDK not installed. Please install it first."
        )

    # Use Unix-style relative path for SDK compatibility
    data_path = data_path.replace('\\', '/')
    if not data_path.endswith('/'):
        data_path = data_path + '/'

    print("Loading CMU-MOSI data...")

    # Load high-level features (pre-extracted features for all modalities)
    print("  Loading high-level features...")
    data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, data_path)
    data.add_computational_sequences(mmdatasdk.cmu_mosi.labels, data_path)

    # Align all sequences to opinion segments
    print("  Aligning all modalities to opinion segments...")
    data.align('Opinion Segment Labels')

    # Extract computational sequences (all are aligned)
    text_seq = data.computational_sequences.get('glove_vectors')  # Pre-extracted GloVe features
    audio_seq = data.computational_sequences.get('COVAREP')       # Pre-extracted COVAREP features
    video_seq = data.computational_sequences.get('FACET_4.2')    # Pre-extracted FACET features
    label_seq = data.computational_sequences['Opinion Segment Labels']

    if not all([text_seq, audio_seq, video_seq]):
        raise ValueError(
            "Missing modality data. Available sequences: "
            f"{list(data.computational_sequences.keys())}"
        )

    print("Extracting samples...")
    samples = []
    segment_ids = []

    # Get all segment IDs
    all_segments = list(label_seq.data.keys())

    for seg_id in all_segments:
        try:
            # Extract pre-extracted features for all modalities
            text_features = text_seq.data[seg_id]['features']    # GloVe vectors
            audio_features = audio_seq.data[seg_id]['features']  # COVAREP features
            video_features = video_seq.data[seg_id]['features']  # FACET features
            label = label_seq.data[seg_id]['features'][0]        # Sentiment score

            sample = {
                'text': text_features,   # Pre-extracted GloVe features (seq_len, 300)
                'audio': audio_features, # Pre-extracted COVAREP features (seq_len, 74)
                'video': video_features, # Pre-extracted FACET features (seq_len, 35)
                'label': float(label)
            }

            samples.append(sample)
            segment_ids.append(seg_id)

        except KeyError as e:
            # Skip segments missing some modalities
            # print(f"Skipping segment {seg_id}: missing data ({e})")
            continue
        except Exception as e:
            # Skip any other errors
            # print(f"Skipping segment {seg_id}: error ({e})")
            continue

    print(f"Extracted {len(samples)} samples")

    # Create train/val/test splits
    np.random.seed(42)
    indices = np.random.permutation(len(samples))

    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        'train': (train_idx, len(train_idx)),
        'valid': (val_idx, len(val_idx)),
        'test': (test_idx, len(test_idx))
    }

    # Save preprocessed data
    for split_name, (split_idx, split_size) in splits.items():
        split_samples = [samples[i] for i in split_idx]
        split_ids = [segment_ids[i] for i in split_idx]

        save_path = os.path.join(data_path, f'preprocessed_{split_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'samples': split_samples,
                'segment_ids': split_ids
            }, f)

        print(f"[OK] Saved {split_size} {split_name} samples to {save_path}")

    print("\nPreprocessing complete!")
    print(f"  Train: {splits['train'][1]} samples")
    print(f"  Valid: {splits['valid'][1]} samples")
    print(f"  Test: {splits['test'][1]} samples")


# Convenience function for getting DataLoader
def get_mosi_dataloader(
    split: str = 'train',
    data_path: str = 'data/cmumosi/',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for CMU-MOSI dataset

    Args:
        split: Dataset split ('train', 'valid', or 'test')
        data_path: Path to CMU-MOSI data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    # Normalize to Unix-style path
    data_path = data_path.replace('\\', '/')
    if not data_path.endswith('/'):
        data_path = data_path + '/'
    dataset = MOSIDataset(data_path=data_path, split=split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == '__main__':
    # Example usage
    print("CMU-MOSI Dataset Loader")
    print("=" * 50)

    # Step 1: Download (only needed once)
    print("\n[1] Downloading dataset...")
    download_mosi('data/cmumosi/')

    # Step 2: Preprocess (only needed once)
    print("\n[2] Preprocessing dataset...")
    preprocess_mosi('data/cmumosi/')

    # Step 3: Create DataLoader
    print("\n[3] Creating DataLoader...")
    train_loader = get_mosi_dataloader(split='train', batch_size=4)

    # Step 4: Test loading a batch
    print("\n[4] Testing batch loading...")
    for batch in train_loader:
        print(f"  Text shape: {batch['text'].shape if isinstance(batch['text'], torch.Tensor) else 'N/A'}")
        print(f"  Audio shape: {batch['audio'].shape if isinstance(batch['audio'], torch.Tensor) else 'N/A'}")
        print(f"  Video shape: {batch['video'].shape if isinstance(batch['video'], torch.Tensor) else 'N/A'}")
        print(f"  Segment IDs: {batch['segment_id'][:2]}...")
        break

    print("\n[OK] Dataset ready for training!")
