"""
LAION Dataset Wrangler for Large-Scale Image-Text Training

This script downloads and processes subsets of the LAION-400M/LAION-5B datasets
for training multimodal encoders. LAION provides CLIP-filtered image-text pairs
at massive scale.

Features:
- Streaming download of metadata (no full dataset download needed)
- Image URL validation and filtering
- Balanced sampling across different qualities
- Checkpoint support for interrupted downloads
- Integration with existing training pipeline

Usage:
    # Download 1M sample from LAION-400M
    python scripts/data_wrangling/wrangle_laion_data.py --subset 400m --num-samples 1000000
    
    # Download high-quality subset from LAION-5B 
    python scripts/data_wrangling/wrangle_laion_data.py --subset 5b --min-score 0.3 --num-samples 10000000
    
    # Resume interrupted download
    python scripts/data_wrangling/wrangle_laion_data.py --resume --checkpoint data/laion/checkpoint.pkl
"""

import os
import sys
import json
import pickle
import argparse
import hashlib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LAIONDataWrangler:
    """Download and process LAION image-text pairs for training."""
    
    # LAION metadata URLs (parquet files)
    METADATA_URLS = {
        '400m': [
            'https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
            'https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00001-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet',
            # Add more parts as needed
        ],
        '5b': [
            # LAION-5B is distributed differently, typically through Hugging Face
            # These would be the actual URLs when available
        ],
        '2b-en': [
            # English subset of LAION-5B (2.32B samples)
        ]
    }
    
    # Hugging Face datasets alternative
    HF_DATASETS = {
        '400m': 'laion/laion400m',
        '5b': 'laion/laion5b',
        '2b-en': 'laion/laion2b-en'
    }
    
    def __init__(self,
                 output_dir: str = 'data/laion/',
                 subset: str = '400m',
                 num_samples: int = 1000000,
                 min_score: float = 0.25,
                 min_width: int = 256,
                 min_height: int = 256,
                 max_workers: int = 8,
                 validate_urls: bool = False,
                 use_huggingface: bool = True):
        """
        Initialize LAION data wrangler.
        
        Args:
            output_dir: Directory to save processed data
            subset: Which LAION subset to use ('400m', '5b', '2b-en')
            num_samples: Number of samples to download
            min_score: Minimum CLIP similarity score
            min_width: Minimum image width
            min_height: Minimum image height
            max_workers: Number of parallel workers for validation
            validate_urls: Whether to validate image URLs (slow)
            use_huggingface: Use HuggingFace datasets API instead of direct download
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.subset = subset
        self.num_samples = num_samples
        self.min_score = min_score
        self.min_width = min_width
        self.min_height = min_height
        self.max_workers = max_workers
        self.validate_urls = validate_urls
        self.use_huggingface = use_huggingface
        
        # Checkpoint for resuming
        self.checkpoint_path = self.output_dir / f'checkpoint_{subset}.pkl'
        self.processed_samples = []
        self.processed_ids = set()
        
    def download_metadata(self) -> pd.DataFrame:
        """Download and load LAION metadata."""
        logger.info(f"Downloading LAION-{self.subset} metadata...")
        
        if self.use_huggingface:
            return self._download_from_huggingface()
        else:
            return self._download_from_urls()
    
    def _download_from_huggingface(self) -> pd.DataFrame:
        """Download using HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("Please install datasets: pip install datasets")
            sys.exit(1)
        
        dataset_name = self.HF_DATASETS.get(self.subset)
        if not dataset_name:
            raise ValueError(f"Unknown subset: {self.subset}")
        
        logger.info(f"Loading {dataset_name} from HuggingFace...")
        
        # Stream dataset to avoid downloading everything
        dataset = load_dataset(
            dataset_name,
            split='train',
            streaming=True
        )
        
        # Collect samples
        samples = []
        for i, item in enumerate(tqdm(dataset, total=self.num_samples)):
            if i >= self.num_samples:
                break
            
            # Filter by quality
            if 'similarity' in item and item['similarity'] < self.min_score:
                continue
            if 'width' in item and item['width'] < self.min_width:
                continue
            if 'height' in item and item['height'] < self.min_height:
                continue
            
            samples.append({
                'caption': item.get('caption', item.get('text', '')),
                'url': item.get('url', ''),
                'similarity': item.get('similarity', 0.0),
                'width': item.get('width', 0),
                'height': item.get('height', 0),
                'nsfw': item.get('nsfw', 'UNLIKELY'),
                'watermark': item.get('pwatermark', 0.0)
            })
        
        return pd.DataFrame(samples)
    
    def _download_from_urls(self) -> pd.DataFrame:
        """Download from direct URLs (parquet files)."""
        urls = self.METADATA_URLS.get(self.subset, [])
        if not urls:
            logger.warning(f"No direct URLs for {self.subset}, trying HuggingFace...")
            return self._download_from_huggingface()
        
        all_data = []
        for url in urls:
            logger.info(f"Downloading {url}...")
            try:
                df = pd.read_parquet(url)
                all_data.append(df)
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Failed to download any metadata")
        
        return pd.concat(all_data, ignore_index=True)
    
    def filter_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to samples."""
        logger.info(f"Filtering {len(df)} samples...")
        
        # Apply filters
        if 'similarity' in df.columns:
            df = df[df['similarity'] >= self.min_score]
        if 'width' in df.columns:
            df = df[df['width'] >= self.min_width]
        if 'height' in df.columns:
            df = df[df['height'] >= self.min_height]
        
        # Remove NSFW content
        if 'nsfw' in df.columns:
            df = df[df['nsfw'].isin(['UNLIKELY', 'VERY_UNLIKELY'])]
        
        # Remove high watermark probability
        if 'pwatermark' in df.columns:
            df = df[df['pwatermark'] < 0.5]
        
        logger.info(f"Filtered to {len(df)} samples")
        return df
    
    def validate_url(self, url: str) -> bool:
        """Check if image URL is still valid."""
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    def validate_urls_parallel(self, urls: List[str]) -> List[bool]:
        """Validate multiple URLs in parallel."""
        valid = [False] * len(urls)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.validate_url, url): i 
                for i, url in enumerate(urls)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(urls), desc="Validating URLs"):
                idx = future_to_idx[future]
                try:
                    valid[idx] = future.result()
                except:
                    valid[idx] = False
        
        return valid
    
    def process_samples(self, df: pd.DataFrame) -> List[Dict]:
        """Process and format samples for training."""
        logger.info(f"Processing {len(df)} samples...")
        
        # Sample if we have more than needed
        if len(df) > self.num_samples:
            df = df.sample(n=self.num_samples, random_state=42)
        
        samples = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Create unique ID
            sample_id = hashlib.md5(
                f"{row.get('url', '')}_{row.get('caption', '')}".encode()
            ).hexdigest()
            
            # Skip if already processed (for resuming)
            if sample_id in self.processed_ids:
                continue
            
            sample = {
                'id': sample_id,
                'caption': row.get('caption', row.get('text', '')),
                'url': row.get('url', ''),
                'metadata': {
                    'similarity': float(row.get('similarity', 0.0)),
                    'width': int(row.get('width', 0)),
                    'height': int(row.get('height', 0)),
                    'source': f'laion-{self.subset}'
                }
            }
            
            samples.append(sample)
            self.processed_ids.add(sample_id)
        
        # Validate URLs if requested
        if self.validate_urls and samples:
            logger.info("Validating image URLs...")
            urls = [s['url'] for s in samples]
            valid = self.validate_urls_parallel(urls)
            samples = [s for s, v in zip(samples, valid) if v]
            logger.info(f"{len(samples)} samples have valid URLs")
        
        return samples
    
    def save_checkpoint(self, samples: List[Dict]):
        """Save checkpoint for resuming."""
        checkpoint = {
            'processed_ids': self.processed_ids,
            'samples': samples,
            'config': {
                'subset': self.subset,
                'num_samples': self.num_samples,
                'min_score': self.min_score
            }
        }
        
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Saved checkpoint to {self.checkpoint_path}")
    
    def load_checkpoint(self) -> Optional[List[Dict]]:
        """Load checkpoint if it exists."""
        if not self.checkpoint_path.exists():
            return None
        
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.processed_ids = checkpoint['processed_ids']
        logger.info(f"Resumed with {len(self.processed_ids)} processed samples")
        return checkpoint['samples']
    
    def split_data(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Split data into train/valid/test sets."""
        np.random.seed(42)
        np.random.shuffle(samples)
        
        n_samples = len(samples)
        n_train = int(0.8 * n_samples)
        n_valid = int(0.1 * n_samples)
        
        splits = {
            'train': samples[:n_train],
            'valid': samples[n_train:n_train + n_valid],
            'test': samples[n_train + n_valid:]
        }
        
        logger.info(f"Split sizes - Train: {len(splits['train'])}, "
                   f"Valid: {len(splits['valid'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def save_data(self, samples: List[Dict]):
        """Save processed data to disk."""
        # Split data
        splits = self.split_data(samples)
        
        # Save each split
        for split_name, split_data in splits.items():
            output_path = self.output_dir / f'laion_{self.subset}_{split_name}.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(split_data, f)
            logger.info(f"Saved {len(split_data)} samples to {output_path}")
        
        # Save metadata
        metadata = {
            'subset': self.subset,
            'total_samples': len(samples),
            'min_score': self.min_score,
            'min_width': self.min_width,
            'min_height': self.min_height,
            'splits': {k: len(v) for k, v in splits.items()},
            'date_created': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = self.output_dir / f'laion_{self.subset}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def run(self, resume: bool = False):
        """Run the complete wrangling pipeline."""
        logger.info(f"Starting LAION-{self.subset} data wrangling...")
        
        # Try to resume from checkpoint
        if resume:
            samples = self.load_checkpoint()
            if samples and len(samples) >= self.num_samples:
                logger.info("Checkpoint has enough samples, skipping download")
                self.save_data(samples)
                return
        else:
            samples = []
        
        # Download and process new samples
        df = self.download_metadata()
        df = self.filter_samples(df)
        new_samples = self.process_samples(df)
        
        # Combine with checkpoint
        samples.extend(new_samples)
        
        # Save checkpoint
        self.save_checkpoint(samples)
        
        # Save final data
        self.save_data(samples)
        
        logger.info(f"Successfully processed {len(samples)} LAION samples!")


def main():
    parser = argparse.ArgumentParser(description='Wrangle LAION dataset for training')
    parser.add_argument('--subset', type=str, default='400m',
                       choices=['400m', '5b', '2b-en'],
                       help='LAION subset to use')
    parser.add_argument('--num-samples', type=int, default=1000000,
                       help='Number of samples to download')
    parser.add_argument('--output-dir', type=str, default='data/laion/',
                       help='Output directory')
    parser.add_argument('--min-score', type=float, default=0.25,
                       help='Minimum CLIP similarity score')
    parser.add_argument('--min-width', type=int, default=256,
                       help='Minimum image width')
    parser.add_argument('--min-height', type=int, default=256,
                       help='Minimum image height')
    parser.add_argument('--validate-urls', action='store_true',
                       help='Validate image URLs (slow)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--use-huggingface', action='store_true', default=True,
                       help='Use HuggingFace datasets API')
    
    args = parser.parse_args()
    
    wrangler = LAIONDataWrangler(
        output_dir=args.output_dir,
        subset=args.subset,
        num_samples=args.num_samples,
        min_score=args.min_score,
        min_width=args.min_width,
        min_height=args.min_height,
        max_workers=args.max_workers,
        validate_urls=args.validate_urls,
        use_huggingface=args.use_huggingface
    )
    
    wrangler.run(resume=args.resume)


if __name__ == '__main__':
    main()