"""
CMU-MOSI Dataset Loader for Cross-Modal Contrastive Learning

This module provides the MOSIRawVideoDataset for loading TRUE raw multimodal data
from extracted YouTube videos (audio waveforms + video frames).

Dataset: CMU Multimodal Opinion Sentiment Intensity (MOSI)
- 2,199 opinion video segments from YouTube
- Modalities: Text transcripts, Audio waveforms (16kHz), Video frames (RGB)
- Labels: Sentiment intensity scores

Usage:
    from Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset, download_mosi

    # Download MOSI metadata (first time only)
    download_mosi('data/cmumosi/mosi/')

    # Load raw video dataset
    dataset = MOSIRawVideoDataset(
        split='train',
        mosi_data_path='data/cmumosi/mosi/',
        audio_dir='data/cmumosi/audio/',
        video_dir='data/cmumosi/frames/'
    )

    # Use with DataLoader and lazy loading collate function
    from torch.utils.data import DataLoader
    from Training.train_raw_encoders import collate_fn_raw_video

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_raw_video)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict
import h5py


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


class MOSIRawVideoDataset(Dataset):
    """
    PyTorch Dataset for CMU-MOSI TRUE raw multimodal data from extracted videos.

    This dataset loads:
    - Raw text transcripts (strings from MOSI words sequence)
    - Paths to extracted audio files (.wav at 16kHz for Whisper)
    - Paths to extracted video frames (.jpg for ViT)
    - Sentiment labels from Opinion_Labels

    This is the RECOMMENDED approach for training encoders on raw modalities.

    Usage:
        dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/'
        )

        sample = dataset[0]
        # Returns:
        # {
        #     'text': "I really enjoyed the movie",
        #     'audio': 'data/cmumosi/audio/BvYR0L6f2Ig.wav',
        #     'video': 'data/cmumosi/frames/BvYR0L6f2Ig.jpg',
        #     'label': 2.5,
        #     'segment_id': 'BvYR0L6f2Ig'
        # }
    """

    def __init__(
        self,
        split: str = 'train',
        mosi_data_path: str = 'data/cmumosi/mosi/',
        audio_dir: str = 'data/cmumosi/audio/',
        video_dir: str = 'data/cmumosi/frames/',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Args:
            split: Dataset split ('train', 'valid', or 'test')
            mosi_data_path: Path to MOSI dataset (for loading raw data via SDK)
            audio_dir: Directory containing extracted audio files
            video_dir: Directory containing extracted video frames
            train_ratio: Proportion for training split (default: 0.7)
            val_ratio: Proportion for validation split (default: 0.15)
            seed: Random seed for reproducible splits (default: 42)
        """
        super().__init__()

        self.split = split
        self.mosi_data_path = mosi_data_path.replace('\\', '/')
        if not self.mosi_data_path.endswith('/'):
            self.mosi_data_path = self.mosi_data_path + '/'
        self.audio_dir = audio_dir
        self.video_dir = video_dir

        # Import MOSI SDK
        try:
            from mmsdk import mmdatasdk
        except ImportError:
            raise ImportError(
                "CMU-MultimodalSDK not installed. Install it:\n"
                "git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git\n"
                "cd CMU-MultimodalSDK && pip install ."
            )

        print(f"Loading raw MOSI data from {self.mosi_data_path}...")

        # Load raw data (for text transcripts)
        print("  Loading words sequence...")
        raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, self.mosi_data_path)

        # Load high-level features (for labels)
        print("  Loading opinion labels...")
        highlevel_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, self.mosi_data_path)
        highlevel_data.add_computational_sequences(mmdatasdk.cmu_mosi.labels, self.mosi_data_path)

        # Extract sequences
        words_seq = raw_data.computational_sequences['words']
        labels_seq = highlevel_data.computational_sequences['Opinion Segment Labels']

        # Get all segment IDs that have both text and labels
        all_segment_ids = []
        for seg_id in words_seq.data.keys():
            if seg_id in labels_seq.data:
                all_segment_ids.append(seg_id)

        print(f"  Found {len(all_segment_ids)} segments with text and labels")

        # Create train/valid/test splits based on video IDs
        # This ensures segments from the same video stay in the same split
        video_ids_to_segments = {}
        for seg_id in all_segment_ids:
            video_id = seg_id.split('[')[0]
            if video_id not in video_ids_to_segments:
                video_ids_to_segments[video_id] = []
            video_ids_to_segments[video_id].append(seg_id)

        # Shuffle video IDs for random split
        video_ids = list(video_ids_to_segments.keys())
        np.random.seed(seed)
        np.random.shuffle(video_ids)

        # Split video IDs
        n_train = int(len(video_ids) * train_ratio)
        n_val = int(len(video_ids) * val_ratio)

        train_video_ids = video_ids[:n_train]
        val_video_ids = video_ids[n_train:n_train + n_val]
        test_video_ids = video_ids[n_train + n_val:]

        # Get segment IDs for this split
        if split == 'train':
            split_video_ids = train_video_ids
        elif split == 'valid':
            split_video_ids = val_video_ids
        elif split == 'test':
            split_video_ids = test_video_ids
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'valid', or 'test'")

        # Collect segment IDs for this split
        split_segment_ids = []
        for video_id in split_video_ids:
            split_segment_ids.extend(video_ids_to_segments[video_id])

        print(f"  {split.capitalize()} split: {len(split_segment_ids)} segments from {len(split_video_ids)} videos")

        # Build samples list
        self.samples = []
        skipped_no_audio = 0
        skipped_no_video = 0
        skipped_no_data = 0

        for seg_id in split_segment_ids:
            try:
                # Parse segment ID (format: either 'video_id[num]' or just 'video_id')
                if '[' in seg_id:
                    video_id = seg_id.split('[')[0]
                    segment_num = int(seg_id.split('[')[1].rstrip(']'))
                    file_basename = f"{video_id}_seg{segment_num}"
                else:
                    # Raw MOSI segments don't have brackets - use segment ID directly
                    video_id = seg_id[:11] if len(seg_id) >= 11 else seg_id
                    file_basename = seg_id.replace('[', '_').replace(']', '')

                # Check if audio and video files exist
                audio_path = os.path.join(self.audio_dir, f"{file_basename}.wav")
                video_path = os.path.join(self.video_dir, f"{file_basename}.jpg")

                if not os.path.exists(audio_path):
                    skipped_no_audio += 1
                    continue

                if not os.path.exists(video_path):
                    skipped_no_video += 1
                    continue

                # Extract text transcript
                words_data = words_seq.data[seg_id]
                features = words_data['features']

                # Handle HDF5 dataset objects
                if isinstance(features, h5py.Dataset):
                    # Read HDF5 dataset as array
                    features = features[:]

                # Convert to text
                if isinstance(features, (list, np.ndarray)):
                    # Decode bytes if needed
                    if len(features) > 0 and isinstance(features[0], bytes):
                        text = ' '.join([f.decode('utf-8') if isinstance(f, bytes) else str(f) for f in features.flatten()])
                    else:
                        text = ' '.join([str(w) for w in features.flatten()])
                else:
                    text = str(features)

                # Extract label
                label = float(labels_seq.data[seg_id]['features'][0])

                self.samples.append({
                    'text': text,
                    'audio': audio_path,
                    'video': video_path,
                    'label': label,
                    'segment_id': seg_id
                })

            except KeyError:
                skipped_no_data += 1
                continue
            except Exception as e:
                skipped_no_data += 1
                continue

        print(f"[OK] Loaded {len(self.samples)} samples for {split} split")
        if skipped_no_audio > 0:
            print(f"  [INFO] Skipped {skipped_no_audio} segments (no audio file)")
        if skipped_no_video > 0:
            print(f"  [INFO] Skipped {skipped_no_video} segments (no video frame)")
        if skipped_no_data > 0:
            print(f"  [INFO] Skipped {skipped_no_data} segments (missing data)")

        if len(self.samples) > 0:
            print(f"  Sample text: '{self.samples[0]['text'][:60]}...'")
            print(f"  Sample audio: {os.path.basename(self.samples[0]['audio'])}")
            print(f"  Sample video: {os.path.basename(self.samples[0]['video'])}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns dictionary with raw multimodal data for one segment.

        Audio and video are returned as PATHS (lazy loading).
        Actual loading happens in the collate_fn.

        Returns:
            dict with keys:
                - 'text': Raw text transcript (str)
                - 'audio': Path to audio file (str)
                - 'video': Path to video frame (str)
                - 'label': Sentiment intensity (float)
                - 'segment_id': Unique segment identifier (str)
        """
        return self.samples[idx]


if __name__ == '__main__':
    # Example usage
    print("CMU-MOSI Raw Video Dataset Loader")
    print("=" * 50)

    # Step 1: Download (only needed once)
    print("\n[1] Downloading dataset...")
    download_mosi('data/cmumosi/mosi/')

    # Step 2: Load dataset
    print("\n[2] Loading dataset...")
    dataset = MOSIRawVideoDataset(
        split='train',
        mosi_data_path='data/cmumosi/mosi/',
        audio_dir='data/cmumosi/audio/',
        video_dir='data/cmumosi/frames/'
    )

    print(f"\n[OK] Dataset ready with {len(dataset)} samples!")
