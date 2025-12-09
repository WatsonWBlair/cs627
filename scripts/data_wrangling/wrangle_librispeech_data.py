"""
LibriSpeech Dataset Wrangler for Audio-Text Training

This script downloads and processes the LibriSpeech ASR corpus for training
audio encoders. LibriSpeech contains ~1000 hours of 16kHz English speech
derived from audiobooks.

Features:
- Download clean and other speech subsets
- Extract audio files and transcripts
- Create audio-text pairs for contrastive learning
- Support for different dataset sizes (100h, 360h, 500h)
- Audio preprocessing and normalization

Usage:
    # Download clean-100 subset
    python scripts/data_wrangling/wrangle_librispeech_data.py --subset clean-100
    
    # Download full clean dataset (360h)
    python scripts/data_wrangling/wrangle_librispeech_data.py --subset clean-360
    
    # Download both clean and other datasets
    python scripts/data_wrangling/wrangle_librispeech_data.py --subset all
"""

import os
import sys
import json
import pickle
import argparse
import tarfile
import hashlib
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LibriSpeechWrangler:
    """Download and process LibriSpeech dataset for audio-text training."""
    
    # LibriSpeech download URLs from OpenSLR
    DATASET_URLS = {
        'dev-clean': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
        'dev-other': 'https://www.openslr.org/resources/12/dev-other.tar.gz',
        'test-clean': 'https://www.openslr.org/resources/12/test-clean.tar.gz',
        'test-other': 'https://www.openslr.org/resources/12/test-other.tar.gz',
        'train-clean-100': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
        'train-clean-360': 'https://www.openslr.org/resources/12/train-clean-360.tar.gz',
        'train-other-500': 'https://www.openslr.org/resources/12/train-other-500.tar.gz'
    }
    
    # Dataset sizes (approximate)
    DATASET_SIZES = {
        'dev-clean': '337MB',
        'dev-other': '314MB',
        'test-clean': '346MB',
        'test-other': '328MB',
        'train-clean-100': '6.3GB',
        'train-clean-360': '23GB',
        'train-other-500': '30GB'
    }
    
    # Subset mappings
    SUBSET_MAPPING = {
        'clean-100': ['train-clean-100', 'dev-clean', 'test-clean'],
        'clean-360': ['train-clean-360', 'dev-clean', 'test-clean'],
        'other-500': ['train-other-500', 'dev-other', 'test-other'],
        'clean-all': ['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean'],
        'all': ['train-clean-100', 'train-clean-360', 'train-other-500',
                'dev-clean', 'dev-other', 'test-clean', 'test-other']
    }
    
    def __init__(self,
                 output_dir: str = 'data/librispeech/',
                 subset: str = 'clean-100',
                 sample_rate: int = 16000,
                 max_duration: float = 20.0,
                 min_duration: float = 0.5,
                 normalize_audio: bool = True,
                 max_samples: Optional[int] = None):
        """
        Initialize LibriSpeech wrangler.
        
        Args:
            output_dir: Directory to save processed data
            subset: Which subset to download
            sample_rate: Target sample rate for audio
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            normalize_audio: Whether to normalize audio amplitude
            max_samples: Maximum number of samples to process
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.subset = subset
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize_audio = normalize_audio
        self.max_samples = max_samples
        
        # Get datasets to download
        self.datasets_to_download = self.SUBSET_MAPPING.get(subset, [subset])
        
        # Paths
        self.raw_dir = self.output_dir / 'LibriSpeech'
        self.cache_dir = self.output_dir / 'cache'
        self.processed_dir = self.output_dir / 'processed'
        
        for dir_path in [self.cache_dir, self.processed_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def download_file(self, url: str, dest_path: Path) -> Path:
        """Download file with progress bar."""
        if dest_path.exists():
            logger.info(f"File already exists: {dest_path}")
            return dest_path
        
        logger.info(f"Downloading {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded to {dest_path}")
        return dest_path
    
    def extract_tar(self, tar_path: Path):
        """Extract tar.gz file."""
        logger.info(f"Extracting {tar_path}...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting"):
                tar.extract(member, self.output_dir)
        
        logger.info(f"Extracted to {self.output_dir}")
    
    def download_and_extract(self, dataset_name: str):
        """Download and extract a LibriSpeech subset."""
        if dataset_name not in self.DATASET_URLS:
            logger.warning(f"Unknown dataset: {dataset_name}")
            return
        
        # Check if already extracted
        dataset_dir = self.raw_dir / dataset_name
        if dataset_dir.exists():
            logger.info(f"Dataset already extracted: {dataset_dir}")
            return
        
        # Download
        url = self.DATASET_URLS[dataset_name]
        tar_path = self.cache_dir / f'{dataset_name}.tar.gz'
        
        logger.info(f"Downloading {dataset_name} ({self.DATASET_SIZES.get(dataset_name, 'unknown size')})")
        self.download_file(url, tar_path)
        
        # Extract
        self.extract_tar(tar_path)
    
    def read_transcript(self, trans_file: Path) -> Dict[str, str]:
        """Read transcript file and return utterance ID to text mapping."""
        transcripts = {}
        
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text.upper()  # LibriSpeech convention
        
        return transcripts
    
    def process_audio(self, audio_path: Path) -> Optional[np.ndarray]:
        """Load and process audio file."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Check duration
            duration = len(audio) / sr
            if duration < self.min_duration or duration > self.max_duration:
                return None
            
            # Normalize if requested
            if self.normalize_audio:
                # Peak normalization
                max_val = np.abs(audio).max()
                if max_val > 0:
                    audio = audio / max_val
            
            return audio
            
        except Exception as e:
            logger.warning(f"Failed to process {audio_path}: {e}")
            return None
    
    def process_dataset(self, dataset_name: str) -> List[Dict]:
        """Process a single LibriSpeech dataset."""
        dataset_dir = self.raw_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return []
        
        logger.info(f"Processing {dataset_name}...")
        samples = []
        
        # Find all transcript files
        trans_files = list(dataset_dir.rglob('*.trans.txt'))
        
        for trans_file in tqdm(trans_files, desc=f"Processing {dataset_name}"):
            # Read transcripts
            transcripts = self.read_transcript(trans_file)
            
            # Process each utterance
            for utt_id, text in transcripts.items():
                # Find corresponding audio file
                audio_file = trans_file.parent / f'{utt_id}.flac'
                
                if not audio_file.exists():
                    logger.warning(f"Audio file not found: {audio_file}")
                    continue
                
                # Process audio
                audio_data = self.process_audio(audio_file)
                if audio_data is None:
                    continue
                
                # Create sample
                sample = {
                    'id': utt_id,
                    'text': text,
                    'audio_path': str(audio_file.relative_to(self.output_dir)),
                    'audio_data': audio_data,  # Store processed audio
                    'duration': len(audio_data) / self.sample_rate,
                    'dataset': dataset_name,
                    'metadata': {
                        'speaker_id': utt_id.split('-')[0],
                        'chapter_id': '-'.join(utt_id.split('-')[:2]),
                        'sample_rate': self.sample_rate
                    }
                }
                
                samples.append(sample)
                
                # Check max samples
                if self.max_samples and len(samples) >= self.max_samples:
                    break
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        logger.info(f"Processed {len(samples)} samples from {dataset_name}")
        return samples
    
    def determine_split(self, dataset_name: str) -> str:
        """Determine train/valid/test split based on dataset name."""
        if 'train' in dataset_name:
            return 'train'
        elif 'dev' in dataset_name:
            return 'valid'
        elif 'test' in dataset_name:
            return 'test'
        else:
            return 'train'
    
    def organize_by_splits(self, all_samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize samples by train/valid/test splits."""
        splits = {'train': [], 'valid': [], 'test': []}
        
        for sample in all_samples:
            split = self.determine_split(sample['dataset'])
            splits[split].append(sample)
        
        # If no validation set, create from training
        if not splits['valid'] and splits['train']:
            np.random.seed(42)
            np.random.shuffle(splits['train'])
            n_valid = min(1000, len(splits['train']) // 10)
            splits['valid'] = splits['train'][:n_valid]
            splits['train'] = splits['train'][n_valid:]
        
        logger.info(f"Split sizes - Train: {len(splits['train'])}, "
                   f"Valid: {len(splits['valid'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def save_data(self, splits: Dict[str, List[Dict]]):
        """Save processed data to disk."""
        for split_name, samples in splits.items():
            if not samples:
                continue
            
            # Separate audio data from metadata
            audio_data = {s['id']: s.pop('audio_data') for s in samples}
            
            # Save metadata
            metadata_path = self.processed_dir / f'librispeech_{split_name}_metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(samples, f)
            logger.info(f"Saved {len(samples)} metadata entries to {metadata_path}")
            
            # Save audio data separately (large!)
            audio_path = self.processed_dir / f'librispeech_{split_name}_audio.npz'
            np.savez_compressed(audio_path, **audio_data)
            logger.info(f"Saved audio data to {audio_path}")
        
        # Save dataset info
        info = {
            'subset': self.subset,
            'datasets': self.datasets_to_download,
            'sample_rate': self.sample_rate,
            'normalize_audio': self.normalize_audio,
            'total_samples': sum(len(s) for s in splits.values()),
            'splits': {k: len(v) for k, v in splits.items()},
            'max_duration': self.max_duration,
            'min_duration': self.min_duration
        }
        
        info_path = self.processed_dir / 'librispeech_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved dataset info to {info_path}")
    
    def create_text_audio_pairs(self, splits: Dict[str, List[Dict]]):
        """Create text-audio training pairs for contrastive learning."""
        for split_name, samples in splits.items():
            if not samples:
                continue
            
            pairs = []
            for sample in samples:
                pair = {
                    'id': sample['id'],
                    'text': sample['text'],
                    'audio_path': sample['audio_path'],
                    'duration': sample['duration'],
                    'metadata': sample['metadata']
                }
                pairs.append(pair)
            
            # Save pairs
            pairs_path = self.processed_dir / f'librispeech_{split_name}_pairs.pkl'
            with open(pairs_path, 'wb') as f:
                pickle.dump(pairs, f)
            logger.info(f"Saved {len(pairs)} text-audio pairs to {pairs_path}")
    
    def run(self):
        """Run the complete wrangling pipeline."""
        logger.info(f"Starting LibriSpeech data wrangling (subset: {self.subset})...")
        
        # Download and extract datasets
        for dataset_name in self.datasets_to_download:
            self.download_and_extract(dataset_name)
        
        # Process all datasets
        all_samples = []
        for dataset_name in self.datasets_to_download:
            samples = self.process_dataset(dataset_name)
            all_samples.extend(samples)
            
            if self.max_samples and len(all_samples) >= self.max_samples:
                all_samples = all_samples[:self.max_samples]
                break
        
        if not all_samples:
            logger.error("No samples processed!")
            return
        
        # Organize by splits
        splits = self.organize_by_splits(all_samples)
        
        # Save data
        self.save_data(splits)
        
        # Create training pairs
        self.create_text_audio_pairs(splits)
        
        logger.info(f"Successfully processed LibriSpeech dataset!")
        logger.info(f"Total samples: {len(all_samples)}")


def main():
    parser = argparse.ArgumentParser(description='Wrangle LibriSpeech dataset')
    parser.add_argument('--subset', type=str, default='clean-100',
                       choices=['clean-100', 'clean-360', 'other-500', 
                               'clean-all', 'all'] + list(LibriSpeechWrangler.DATASET_URLS.keys()),
                       help='Which subset to download')
    parser.add_argument('--output-dir', type=str, default='data/librispeech/',
                       help='Output directory')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Target audio sample rate')
    parser.add_argument('--max-duration', type=float, default=20.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--min-duration', type=float, default=0.5,
                       help='Minimum audio duration in seconds')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Do not normalize audio amplitude')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    wrangler = LibriSpeechWrangler(
        output_dir=args.output_dir,
        subset=args.subset,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        normalize_audio=not args.no_normalize,
        max_samples=args.max_samples
    )
    
    wrangler.run()


if __name__ == '__main__':
    main()