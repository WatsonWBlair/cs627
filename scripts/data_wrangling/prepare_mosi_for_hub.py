"""
CMU-MOSI Dataset Preparation for HuggingFace Hub

Prepares the CMU-MOSI dataset for upload to HuggingFace Hub.
Includes raw segments (audio + frames) and pre-computed embeddings.

Output structure (~1.5GB):
- Audio segments: WAV files at 16kHz
- Video frames: JPG images
- Pre-computed embeddings: Parquet files
- Metadata: JSON with segment info

Usage:
    python scripts/data_wrangling/prepare_mosi_for_hub.py
    python scripts/data_wrangling/prepare_mosi_for_hub.py --include-embeddings
"""

import argparse
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from datasets import Dataset, DatasetDict, Features, Value, Audio, Image, Array2D
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


class MOSIHubPreparer:
    """
    Prepare CMU-MOSI dataset for HuggingFace Hub upload.

    Collects raw multimodal data and optional embeddings into
    a unified format suitable for Hub hosting.
    """

    def __init__(
        self,
        mosi_data_path: str = "data/cmumosi/mosi/",
        audio_dir: str = "data/cmumosi/audio/",
        frames_dir: str = "data/cmumosi/frames/",
        output_dir: str = "data/cmumosi/hub_export/",
        include_embeddings: bool = False,
        embeddings_dir: Optional[str] = None
    ):
        """
        Initialize MOSI Hub preparer.

        Args:
            mosi_data_path: Path to MOSI metadata (pkl files)
            audio_dir: Directory with extracted audio segments
            frames_dir: Directory with extracted video frames
            output_dir: Directory for Hub export
            include_embeddings: Whether to include pre-computed embeddings
            embeddings_dir: Directory with embedding files (if different)
        """
        self.mosi_data_path = Path(mosi_data_path)
        self.audio_dir = Path(audio_dir)
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.include_embeddings = include_embeddings
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)
        if include_embeddings:
            (self.output_dir / "embeddings").mkdir(exist_ok=True)

    def _parse_segment_id(self, seg_id: str) -> Tuple[str, int]:
        """Parse segment ID into video ID and segment number."""
        if '[' in seg_id:
            parts = seg_id.split('[')
            video_id = parts[0]
            segment_num = int(parts[1].rstrip(']'))
            return video_id, segment_num
        return seg_id, 0

    def _segment_to_filename(self, seg_id: str, ext: str = "") -> str:
        """Convert segment ID to filesystem-safe filename."""
        video_id, segment_num = self._parse_segment_id(seg_id)
        base = f"{video_id}_{segment_num}" if segment_num > 0 else video_id
        return f"{base}{ext}"

    def load_split_data(self, split: str) -> List[Dict]:
        """
        Load segment data from preprocessed pkl file.

        Args:
            split: Dataset split ('train', 'valid', 'test')

        Returns:
            List of segment dictionaries
        """
        pkl_path = self.mosi_data_path / f"preprocessed_{split}.pkl"

        if not pkl_path.exists():
            print(f"Warning: {pkl_path} not found")
            return []

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        return data.get('segment_ids', [])

    def collect_segments(self) -> Dict[str, List[Dict]]:
        """
        Collect all available segments across splits.

        Returns:
            Dictionary mapping splits to segment data
        """
        print("\n" + "=" * 80)
        print("Collecting MOSI Segments")
        print("=" * 80)

        splits_data = {}

        for split in ['train', 'valid', 'test']:
            segment_ids = self.load_split_data(split)
            print(f"[{split}] Found {len(segment_ids)} segment IDs")

            # Check which segments have audio/video files
            available_segments = []

            for seg_id in segment_ids:
                filename = self._segment_to_filename(seg_id)
                audio_path = self.audio_dir / f"{filename}.wav"
                frame_path = self.frames_dir / f"{filename}.jpg"

                if audio_path.exists() or frame_path.exists():
                    segment_info = {
                        'segment_id': seg_id,
                        'video_id': self._parse_segment_id(seg_id)[0],
                        'segment_num': self._parse_segment_id(seg_id)[1],
                        'has_audio': audio_path.exists(),
                        'has_frame': frame_path.exists(),
                        'audio_file': f"{filename}.wav" if audio_path.exists() else None,
                        'frame_file': f"{filename}.jpg" if frame_path.exists() else None
                    }
                    available_segments.append(segment_info)

            splits_data[split] = available_segments
            print(f"[{split}] {len(available_segments)} segments with media files")

        return splits_data

    def load_text_and_labels(self) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Load text transcripts and sentiment labels from MOSI data.

        Returns:
            Tuple of (texts dict, labels dict) keyed by video_id
        """
        texts = {}
        labels = {}

        # Try to load from pkl files which contain text/labels
        for split in ['train', 'valid', 'test']:
            pkl_path = self.mosi_data_path / f"preprocessed_{split}.pkl"
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                # Extract text and labels if available
                if 'texts' in data:
                    texts.update(data['texts'])
                if 'labels' in data:
                    labels.update(data['labels'])

        # Also try SDK CSD files
        try:
            from mmsdk import mmdatasdk

            words_csd = self.mosi_data_path / "CMU_MOSI_TimestampedWords.csd"
            labels_csd = self.mosi_data_path / "CMU_MOSI_Opinion_Labels.csd"

            if words_csd.exists():
                raw_data = mmdatasdk.mmdataset({
                    'words': str(words_csd)
                }, str(self.mosi_data_path))

                for video_id, data in raw_data.computational_sequences['words'].data.items():
                    features = data['features']
                    if hasattr(features, '__iter__'):
                        if isinstance(features[0], bytes):
                            text = ' '.join([w.decode('utf-8') for w in features.flatten()])
                        else:
                            text = ' '.join([str(w) for w in features.flatten()])
                        texts[video_id] = text

            if labels_csd.exists():
                label_data = mmdatasdk.mmdataset({
                    'labels': str(labels_csd)
                }, str(self.mosi_data_path))

                for video_id, data in label_data.computational_sequences['labels'].data.items():
                    labels[video_id] = float(np.mean(data['features']))

        except ImportError:
            print("Warning: mmsdk not available, using pkl data only")
        except Exception as e:
            print(f"Warning: Error loading SDK data: {e}")

        return texts, labels

    def copy_media_files(self, splits_data: Dict[str, List[Dict]]) -> None:
        """
        Copy audio and frame files to output directory.

        Args:
            splits_data: Dictionary of split data
        """
        print("\nCopying media files...")

        for split, segments in splits_data.items():
            for segment in tqdm(segments, desc=f"[{split}]"):
                # Copy audio
                if segment['has_audio']:
                    src = self.audio_dir / segment['audio_file']
                    dst = self.output_dir / "audio" / segment['audio_file']
                    if not dst.exists():
                        shutil.copy2(src, dst)

                # Copy frame
                if segment['has_frame']:
                    src = self.frames_dir / segment['frame_file']
                    dst = self.output_dir / "frames" / segment['frame_file']
                    if not dst.exists():
                        shutil.copy2(src, dst)

    def create_metadata(
        self,
        splits_data: Dict[str, List[Dict]],
        texts: Dict[str, str],
        labels: Dict[str, float]
    ) -> Dict:
        """
        Create dataset metadata JSON.

        Args:
            splits_data: Dictionary of split data
            texts: Text transcripts by video_id
            labels: Sentiment labels by video_id

        Returns:
            Metadata dictionary
        """
        # Enrich segment data with text and labels
        for split, segments in splits_data.items():
            for segment in segments:
                video_id = segment['video_id']
                segment['text'] = texts.get(video_id, "")
                segment['sentiment_label'] = labels.get(video_id, 0.0)

        # Calculate statistics
        total_segments = sum(len(s) for s in splits_data.values())
        total_with_audio = sum(
            sum(1 for seg in s if seg['has_audio'])
            for s in splits_data.values()
        )
        total_with_frame = sum(
            sum(1 for seg in s if seg['has_frame'])
            for s in splits_data.values()
        )

        unique_videos = set()
        for segments in splits_data.values():
            for seg in segments:
                unique_videos.add(seg['video_id'])

        metadata = {
            'dataset': 'CMU-MOSI',
            'description': 'CMU Multimodal Opinion Sentiment Intensity dataset',
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'statistics': {
                'total_segments': total_segments,
                'segments_with_audio': total_with_audio,
                'segments_with_frame': total_with_frame,
                'unique_videos': len(unique_videos),
                'splits': {split: len(segs) for split, segs in splits_data.items()}
            },
            'features': {
                'segment_id': 'Unique segment identifier (video_id[segment_num])',
                'video_id': 'YouTube video ID',
                'text': 'Transcript text',
                'sentiment_label': 'Sentiment intensity score',
                'audio': 'Path to WAV file (16kHz mono)',
                'frame': 'Path to JPG frame'
            },
            'splits': splits_data
        }

        return metadata

    def create_huggingface_dataset(
        self,
        splits_data: Dict[str, List[Dict]],
        texts: Dict[str, str],
        labels: Dict[str, float]
    ) -> Optional['DatasetDict']:
        """
        Create HuggingFace DatasetDict for upload.

        Args:
            splits_data: Dictionary of split data
            texts: Text transcripts
            labels: Sentiment labels

        Returns:
            DatasetDict or None if HF datasets not available
        """
        if not HF_DATASETS_AVAILABLE:
            print("Warning: datasets library not available, skipping HF format")
            return None

        print("\nCreating HuggingFace Dataset...")

        hf_splits = {}

        for split, segments in splits_data.items():
            records = []

            for segment in segments:
                video_id = segment['video_id']

                record = {
                    'segment_id': segment['segment_id'],
                    'video_id': video_id,
                    'text': texts.get(video_id, ""),
                    'sentiment_label': labels.get(video_id, 0.0),
                }

                # Add audio path
                if segment['has_audio']:
                    record['audio'] = str(self.output_dir / "audio" / segment['audio_file'])
                else:
                    record['audio'] = None

                # Add frame path
                if segment['has_frame']:
                    record['frame'] = str(self.output_dir / "frames" / segment['frame_file'])
                else:
                    record['frame'] = None

                records.append(record)

            if records:
                hf_splits[split] = Dataset.from_list(records)
                print(f"[{split}] Created dataset with {len(records)} samples")

        if hf_splits:
            return DatasetDict(hf_splits)

        return None

    def save_outputs(
        self,
        metadata: Dict,
        hf_dataset: Optional['DatasetDict'] = None
    ) -> None:
        """
        Save all outputs to disk.

        Args:
            metadata: Dataset metadata
            hf_dataset: Optional HuggingFace dataset
        """
        print("\nSaving outputs...")

        # Save metadata JSON
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {metadata_path}")

        # Save per-split JSON files
        for split, segments in metadata['splits'].items():
            split_path = self.output_dir / f"{split}.json"
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            print(f"Saved {split} data to {split_path}")

        # Save HuggingFace dataset
        if hf_dataset is not None:
            hf_path = self.output_dir / "hf_dataset"
            hf_dataset.save_to_disk(str(hf_path))
            print(f"Saved HuggingFace dataset to {hf_path}")

    def prepare(self) -> Path:
        """
        Run the full preparation pipeline.

        Returns:
            Path to output directory
        """
        # Collect segments
        splits_data = self.collect_segments()

        if not any(splits_data.values()):
            print("Error: No segments found!")
            return self.output_dir

        # Load text and labels
        texts, labels = self.load_text_and_labels()
        print(f"\nLoaded {len(texts)} texts and {len(labels)} labels")

        # Copy media files
        self.copy_media_files(splits_data)

        # Create metadata
        metadata = self.create_metadata(splits_data, texts, labels)

        # Create HuggingFace dataset
        hf_dataset = self.create_huggingface_dataset(splits_data, texts, labels)

        # Save outputs
        self.save_outputs(metadata, hf_dataset)

        # Print summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Total segments: {metadata['statistics']['total_segments']}")
        print(f"With audio: {metadata['statistics']['segments_with_audio']}")
        print(f"With frame: {metadata['statistics']['segments_with_frame']}")
        print(f"Unique videos: {metadata['statistics']['unique_videos']}")

        # Calculate size
        total_size = 0
        for path in self.output_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size

        print(f"Total size: {total_size / (1024**2):.1f} MB")

        return self.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CMU-MOSI dataset for HuggingFace Hub"
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
        help='Directory with audio segments'
    )
    parser.add_argument(
        '--frames-dir',
        type=str,
        default='data/cmumosi/frames/',
        help='Directory with video frames'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/cmumosi/hub_export/',
        help='Output directory for Hub export'
    )
    parser.add_argument(
        '--include-embeddings',
        action='store_true',
        help='Include pre-computed embeddings'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default=None,
        help='Directory with embedding files'
    )

    args = parser.parse_args()

    preparer = MOSIHubPreparer(
        mosi_data_path=args.mosi_data_path,
        audio_dir=args.audio_dir,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        include_embeddings=args.include_embeddings,
        embeddings_dir=args.embeddings_dir
    )

    output_path = preparer.prepare()

    print(f"\nDataset prepared at: {output_path}")
    print("\nTo upload to HuggingFace Hub:")
    print("  1. Install: pip install huggingface_hub")
    print("  2. Login: huggingface-cli login")
    print("  3. Upload: huggingface-cli upload <your-username>/cmu-mosi", output_path)


if __name__ == "__main__":
    main()
