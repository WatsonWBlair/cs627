"""
CMU-MOSI Dataset Loader for Cross-Modal Contrastive Learning

This module provides the MOSIRawVideoDataset for loading multimodal data
from pre-extracted and segmented YouTube videos.

Dataset: CMU Multimodal Opinion Sentiment Intensity (MOSI)
- 2,199 opinion video segments from YouTube (3-10 seconds each)
- Modalities: Text transcripts, Audio waveforms (16kHz), Video frames (RGB)
- Labels: Sentiment intensity scores

All data is eagerly loaded into RAM during dataset initialization for fast training.

Environment Variables:
    SKIP_DOWNLOAD=1 (default): Skip SDK downloads, assume data is pre-staged
    SKIP_DOWNLOAD=0: Enable SDK downloads (for data preparation)

Usage:
    from Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset, download_mosi

    # Download MOSI metadata (first time only, or set SKIP_DOWNLOAD=0)
    download_mosi('data/cmumosi/mosi/')

    # Load dataset (SKIP_DOWNLOAD=1 by default, assumes data is pre-staged)
    dataset = MOSIRawVideoDataset(
        split='train',
        mosi_data_path='data/cmumosi/mosi/',
        audio_dir='data/cmumosi/audio/',
        video_dir='data/cmumosi/frames/'
    )

    # Use with DataLoader
    from torch.utils.data import DataLoader
    from Training.train_encoders import collate_fn_raw_video

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_raw_video)
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional
import h5py
import librosa
from PIL import Image as PILImage

# Skip SDK downloads by default - assumes data is pre-staged (Docker/S3 workflow)
SKIP_DOWNLOAD = os.getenv('SKIP_DOWNLOAD', '1') == '1'


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

    This dataset loads segment IDs from preprocessed pkl files which have fine-grained
    opinion segments (1528+ segments), then loads raw multimodal data:
    - Raw text transcripts (strings from MOSI words sequence)
    - Audio waveforms (16kHz mono numpy arrays from .wav files)
    - Video frames (RGB numpy arrays from .jpg files)
    - Sentiment labels from Opinion_Labels

    This is the RECOMMENDED approach for training encoders on segmented data.

    Usage:
        # Full multimodal dataset (default - requires all modalities)
        dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/'
        )

        # Text-only dataset (useful when audio/video not yet extracted)
        dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/',
            required_modalities=['text']  # Only require text
        )

        sample = dataset[0]
        # Returns:
        # {
        #     'text': "I really enjoyed the movie",
        #     'audio': numpy.ndarray(shape=(96000,), dtype=float32) or None,
        #     'video': numpy.ndarray(shape=(480, 640, 3), dtype=uint8) or None,
        #     'label': 2.5,
        #     'segment_id': 'BvYR0L6f2Ig[0]'
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
        seed: int = 42,
        required_modalities: list = None,
        skip_download: Optional[bool] = None
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
            required_modalities: List of modalities that must be present
                                (default: ['text', 'audio', 'video'] for backward compatibility)
                                Options: 'text', 'audio', 'video'
                                Example: ['text'] for text-only training
            skip_download: Skip SDK downloads (default: uses SKIP_DOWNLOAD env var)
                          True = assume data is pre-staged, load from CDF files directly
                          False = use SDK which may trigger downloads
        """
        super().__init__()

        self.split = split
        self.mosi_data_path = mosi_data_path.replace('\\', '/')
        if not self.mosi_data_path.endswith('/'):
            self.mosi_data_path = self.mosi_data_path + '/'
        self.audio_dir = audio_dir
        self.video_dir = video_dir

        # Use global SKIP_DOWNLOAD if not specified
        self.skip_download = skip_download if skip_download is not None else SKIP_DOWNLOAD

        # Default to requiring all modalities for backward compatibility
        if required_modalities is None:
            required_modalities = ['text', 'audio', 'video']
        self.required_modalities = required_modalities

        # Validate modality names
        valid_modalities = {'text', 'audio', 'video'}
        for modality in self.required_modalities:
            if modality not in valid_modalities:
                raise ValueError(f"Invalid modality '{modality}'. Must be one of: {valid_modalities}")

        print(f"Loading raw MOSI data from {self.mosi_data_path}...")
        print(f"  Required modalities: {', '.join(self.required_modalities)}")
        print(f"  Skip download: {self.skip_download}")

        # Try to load from preprocessed pkl files first (fine-grained segments)
        pkl_split = 'valid' if split == 'valid' else split
        pkl_path = os.path.join(self.mosi_data_path, f'preprocessed_{pkl_split}.pkl')

        if os.path.exists(pkl_path):
            print(f"  Loading segment IDs from {pkl_path}...")
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)
            split_segment_ids = pkl_data['segment_ids']
            print(f"  Found {len(split_segment_ids)} fine-grained segments for {split} split")
        else:
            if self.skip_download:
                raise FileNotFoundError(
                    f"Preprocessed data not found: {pkl_path}\n"
                    f"SKIP_DOWNLOAD=1 requires pre-staged data. Either:\n"
                    f"  1. Run data preparation: SKIP_DOWNLOAD=0 python -c \"from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('{self.mosi_data_path}')\"\n"
                    f"  2. Copy pre-staged data from S3 to {self.mosi_data_path}"
                )
            # Fall back to SDK-based segment loading (coarse-grained)
            print(f"  [WARNING] {pkl_path} not found, falling back to SDK-based loading")
            split_segment_ids = self._load_segments_from_sdk(split, train_ratio, val_ratio, seed)

        # Load raw text data from SDK (words sequence) or from pre-staged CDF files
        print("  Loading words sequence for raw text...")
        try:
            from mmsdk import mmdatasdk

            if self.skip_download:
                # Load from pre-staged CSD files without triggering downloads
                # CSD = Computational Sequence Data (CMU-MultimodalSDK format)
                words_csd = os.path.join(self.mosi_data_path, 'CMU_MOSI_TimestampedWords.csd')
                labels_csd = os.path.join(self.mosi_data_path, 'CMU_MOSI_Opinion_Labels.csd')

                if not os.path.exists(words_csd):
                    raise FileNotFoundError(
                        f"Pre-staged CSD file not found: {words_csd}\n"
                        f"SKIP_DOWNLOAD=1 requires pre-staged data."
                    )

                # Load directly from local CSD files (mmdataset with local paths only)
                raw_data = mmdatasdk.mmdataset({
                    'words': words_csd
                }, self.mosi_data_path)
                words_seq = raw_data.computational_sequences['words']

                # Load labels
                highlevel_data = mmdatasdk.mmdataset({
                    'Opinion Segment Labels': labels_csd
                }, self.mosi_data_path)
                labels_seq = highlevel_data.computational_sequences['Opinion Segment Labels']
            else:
                # Original behavior - may trigger downloads
                raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, self.mosi_data_path)
                words_seq = raw_data.computational_sequences['words']

                # Also load labels
                highlevel_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, self.mosi_data_path)
                highlevel_data.add_computational_sequences(mmdatasdk.cmu_mosi.labels, self.mosi_data_path)
                labels_seq = highlevel_data.computational_sequences['Opinion Segment Labels']
        except ImportError:
            raise ImportError(
                "CMU-MultimodalSDK not installed. Install it:\n"
                "git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git\n"
                "cd CMU-MultimodalSDK && pip install ."
            )

        # Build samples list
        self.samples = []
        skipped_no_audio = 0
        skipped_no_video = 0
        skipped_no_text = 0
        skipped_no_data = 0

        for seg_id in split_segment_ids:
            try:
                # File naming: seg_id.replace('[', '_').replace(']', '')
                # e.g., "BvYR0L6f2Ig[0]" -> "BvYR0L6f2Ig_0"
                file_basename = seg_id.replace('[', '_').replace(']', '')

                # Check if audio and video files exist
                audio_path = os.path.join(self.audio_dir, f"{file_basename}.wav")
                video_path = os.path.join(self.video_dir, f"{file_basename}.jpg")

                audio_exists = os.path.exists(audio_path)
                video_exists = os.path.exists(video_path)

                # Skip if required modality is missing
                if 'audio' in self.required_modalities and not audio_exists:
                    skipped_no_audio += 1
                    continue

                if 'video' in self.required_modalities and not video_exists:
                    skipped_no_video += 1
                    continue

                # Load audio data eagerly (into RAM)
                audio_data = None
                if audio_exists:
                    try:
                        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
                        audio_data = waveform  # numpy array
                    except (RuntimeError, OSError, ValueError) as e:
                        print(f"  [WARNING] Failed to load audio {file_basename}: {e}")
                        if 'audio' in self.required_modalities:
                            skipped_no_audio += 1
                            continue

                # Load video frame eagerly (into RAM)
                video_data = None
                if video_exists:
                    try:
                        frame = PILImage.open(video_path).convert('RGB')
                        video_data = np.array(frame)  # numpy array (H, W, 3)
                    except (OSError, ValueError) as e:
                        print(f"  [WARNING] Failed to load frame {file_basename}: {e}")
                        if 'video' in self.required_modalities:
                            skipped_no_video += 1
                            continue

                # Extract text transcript from SDK
                # For fine-grained segments (e.g., "BvYR0L6f2Ig[0]"), we need to get
                # the video-level text and extract the relevant portion
                video_id = seg_id.split('[')[0] if '[' in seg_id else seg_id

                text = None
                label = None

                # Try to get text from words sequence (video-level)
                if video_id in words_seq.data:
                    words_data = words_seq.data[video_id]
                    features = words_data['features']

                    # Handle HDF5 dataset objects
                    if isinstance(features, h5py.Dataset):
                        features = features[:]

                    # Convert to text
                    if isinstance(features, (list, np.ndarray)):
                        if len(features) > 0 and isinstance(features[0], bytes):
                            text = ' '.join([f.decode('utf-8') if isinstance(f, bytes) else str(f) for f in features.flatten()])
                        else:
                            text = ' '.join([str(w) for w in features.flatten()])
                    else:
                        text = str(features)

                # Try to get label (video-level)
                if video_id in labels_seq.data:
                    label_data = labels_seq.data[video_id]['features']
                    if isinstance(label_data, h5py.Dataset):
                        label_data = label_data[:]
                    # Average all labels for this video
                    label = float(np.mean(label_data))

                if text is None:
                    if 'text' in self.required_modalities:
                        skipped_no_text += 1
                        continue
                    text = ""  # Empty text if not required

                if label is None:
                    label = 0.0  # Default label

                self.samples.append({
                    'text': text,
                    'audio': audio_data,  # numpy array or None
                    'video': video_data,  # numpy array (H, W, 3) or None
                    'label': label,
                    'segment_id': seg_id
                })

            except (KeyError, ValueError, AttributeError, IndexError, TypeError, UnicodeDecodeError) as e:
                skipped_no_data += 1
                continue

        print(f"[OK] Loaded {len(self.samples)} samples for {split} split")
        if skipped_no_audio > 0:
            print(f"  [INFO] Skipped {skipped_no_audio} segments (missing required audio)")
        if skipped_no_video > 0:
            print(f"  [INFO] Skipped {skipped_no_video} segments (missing required video)")
        if skipped_no_text > 0:
            print(f"  [INFO] Skipped {skipped_no_text} segments (missing required text)")
        if skipped_no_data > 0:
            print(f"  [INFO] Skipped {skipped_no_data} segments (data parsing error)")

        # Count optional modality availability
        if len(self.samples) > 0:
            if 'audio' not in self.required_modalities:
                audio_available = sum(1 for s in self.samples if s['audio'] is not None)
                print(f"  [INFO] {audio_available}/{len(self.samples)} samples have optional audio")
            if 'video' not in self.required_modalities:
                video_available = sum(1 for s in self.samples if s['video'] is not None)
                print(f"  [INFO] {video_available}/{len(self.samples)} samples have optional video")

        if len(self.samples) > 0:
            print(f"  Sample text: '{self.samples[0]['text'][:60]}...'")
            if self.samples[0]['audio'] is not None:
                print(f"  Sample audio: shape={self.samples[0]['audio'].shape}, dtype={self.samples[0]['audio'].dtype}")
            if self.samples[0]['video'] is not None:
                print(f"  Sample video: shape={self.samples[0]['video'].shape}, dtype={self.samples[0]['video'].dtype}")

    def _load_segments_from_sdk(self, split, train_ratio, val_ratio, seed):
        """
        Fallback: Load segment IDs from SDK (coarse-grained video-level segments).
        """
        from mmsdk import mmdatasdk

        raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, self.mosi_data_path)
        highlevel_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, self.mosi_data_path)
        highlevel_data.add_computational_sequences(mmdatasdk.cmu_mosi.labels, self.mosi_data_path)

        words_seq = raw_data.computational_sequences['words']
        labels_seq = highlevel_data.computational_sequences['Opinion Segment Labels']

        # Get all segment IDs that have both text and labels
        all_segment_ids = []
        for seg_id in words_seq.data.keys():
            if seg_id in labels_seq.data:
                all_segment_ids.append(seg_id)

        print(f"  Found {len(all_segment_ids)} coarse-grained segments")

        # Split by video IDs
        video_ids = list(set([s.split('[')[0] if '[' in s else s for s in all_segment_ids]))
        np.random.seed(seed)
        np.random.shuffle(video_ids)

        n_train = int(len(video_ids) * train_ratio)
        n_val = int(len(video_ids) * val_ratio)

        if split == 'train':
            split_video_ids = set(video_ids[:n_train])
        elif split == 'valid':
            split_video_ids = set(video_ids[n_train:n_train + n_val])
        else:  # test
            split_video_ids = set(video_ids[n_train + n_val:])

        return [s for s in all_segment_ids if (s.split('[')[0] if '[' in s else s) in split_video_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns dictionary with raw multimodal data for one segment.

        All data is loaded into RAM during initialization (eager loading).

        Returns:
            dict with keys:
                - 'text': Raw text transcript (str)
                - 'audio': Audio waveform (numpy array, shape=(samples,)) or None
                - 'video': Video frame (numpy array, shape=(H, W, 3)) or None
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
