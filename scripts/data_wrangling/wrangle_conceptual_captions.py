"""
Conceptual Captions Dataset Wrangler

This script downloads and processes the Conceptual Captions 3M (CC3M) and
Conceptual Captions 12M (CC12M) datasets for training multimodal encoders.

CC3M contains 3.3M image-text pairs with cleaner, more informative captions.
CC12M contains 12M image-text pairs for large-scale pre-training.

Features:
- Download TSV files from Google Research
- Caption cleaning and filtering
- Image URL validation
- Automatic retry for failed downloads
- Integration with training pipeline

Usage:
    # Download CC3M dataset
    python scripts/data_wrangling/wrangle_conceptual_captions.py --dataset cc3m
    
    # Download CC12M dataset with validation
    python scripts/data_wrangling/wrangle_conceptual_captions.py --dataset cc12m --validate-urls
    
    # Download subset for testing
    python scripts/data_wrangling/wrangle_conceptual_captions.py --dataset cc3m --max-samples 100000
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
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConceptualCaptionsWrangler:
    """Download and process Conceptual Captions datasets."""
    
    # Dataset URLs from Google Research
    DATASET_URLS = {
        'cc3m': {
            'train': 'https://storage.googleapis.com/conceptual-captions/Train_GCC-training.tsv',
            'validation': 'https://storage.googleapis.com/conceptual-captions/Validation_GCC-1.1.0-Validation.tsv'
        },
        'cc12m': {
            # CC12M is distributed as TSV files with image URLs and captions
            'train': 'https://storage.googleapis.com/conceptual-12m/cc12m.tsv'
        }
    }
    
    def __init__(self,
                 dataset: str = 'cc3m',
                 output_dir: str = 'data/conceptual_captions/',
                 max_samples: Optional[int] = None,
                 min_caption_length: int = 3,
                 max_caption_length: int = 256,
                 validate_urls: bool = False,
                 max_workers: int = 8,
                 retry_attempts: int = 3):
        """
        Initialize Conceptual Captions wrangler.
        
        Args:
            dataset: Which dataset to download ('cc3m' or 'cc12m')
            output_dir: Directory to save processed data
            max_samples: Maximum number of samples to download (None for all)
            min_caption_length: Minimum caption length in words
            max_caption_length: Maximum caption length in words
            validate_urls: Whether to validate image URLs
            max_workers: Number of parallel workers for validation
            retry_attempts: Number of retry attempts for failed downloads
        """
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_samples = max_samples
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length
        self.validate_urls = validate_urls
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        
        # Cache directory for downloaded TSV files
        self.cache_dir = self.output_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_tsv(self, url: str, cache_name: str) -> str:
        """Download TSV file with caching."""
        cache_path = self.cache_dir / cache_name
        
        if cache_path.exists():
            logger.info(f"Using cached file: {cache_path}")
            return str(cache_path)
        
        logger.info(f"Downloading {url}...")
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(cache_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                logger.info(f"Downloaded to {cache_path}")
                return str(cache_path)
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def load_tsv_data(self, file_path: str) -> pd.DataFrame:
        """Load TSV data into DataFrame."""
        logger.info(f"Loading TSV data from {file_path}...")
        
        # CC3M and CC12M TSV format: caption\turl
        df = pd.read_csv(
            file_path,
            sep='\t',
            names=['caption', 'url'],
            dtype=str,
            on_bad_lines='skip'
        )
        
        # Remove rows with NaN values
        initial_count = len(df)
        df = df.dropna()
        logger.info(f"Loaded {len(df)} samples (dropped {initial_count - len(df)} invalid rows)")
        
        return df
    
    def clean_caption(self, caption: str) -> str:
        """Clean and normalize caption text."""
        # Remove extra whitespace
        caption = ' '.join(caption.split())
        
        # Fix common HTML entities
        caption = caption.replace('&amp;', '&')
        caption = caption.replace('&lt;', '<')
        caption = caption.replace('&gt;', '>')
        caption = caption.replace('&quot;', '"')
        caption = caption.replace('&#39;', "'")
        
        # Remove URLs from captions
        caption = re.sub(r'http\S+|www\.\S+', '', caption)
        
        # Remove extra punctuation
        caption = re.sub(r'([.!?])\1+', r'\1', caption)
        
        # Ensure proper spacing around punctuation
        caption = re.sub(r'\s+([,.!?])', r'\1', caption)
        
        return caption.strip()
    
    def filter_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to samples."""
        logger.info(f"Filtering {len(df)} samples...")
        
        # Clean captions
        df['caption'] = df['caption'].apply(self.clean_caption)
        
        # Filter by caption length
        df['caption_words'] = df['caption'].apply(lambda x: len(x.split()))
        df = df[
            (df['caption_words'] >= self.min_caption_length) &
            (df['caption_words'] <= self.max_caption_length)
        ]
        
        # Remove samples with empty URLs
        df = df[df['url'].str.len() > 0]
        
        # Remove duplicate URLs
        df = df.drop_duplicates(subset=['url'])
        
        # Remove duplicate captions (keep first)
        df = df.drop_duplicates(subset=['caption'])
        
        logger.info(f"Filtered to {len(df)} samples")
        return df
    
    def validate_url(self, url: str) -> bool:
        """Check if image URL is still valid."""
        try:
            response = requests.head(
                url,
                timeout=5,
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            return response.status_code == 200
        except:
            return False
    
    def validate_urls_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate URLs in parallel."""
        if not self.validate_urls:
            return df
        
        logger.info("Validating image URLs...")
        urls = df['url'].tolist()
        valid_mask = [False] * len(urls)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.validate_url, url): i
                for i, url in enumerate(urls)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(urls)):
                idx = future_to_idx[future]
                try:
                    valid_mask[idx] = future.result()
                except:
                    valid_mask[idx] = False
        
        df_valid = df[valid_mask]
        logger.info(f"{len(df_valid)} samples have valid URLs ({len(df_valid)/len(df)*100:.1f}%)")
        
        return df_valid
    
    def process_samples(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to training format."""
        logger.info("Processing samples...")
        
        # Limit samples if specified
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=42)
        
        samples = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Create unique ID
            sample_id = hashlib.md5(
                f"{row['url']}_{row['caption']}".encode()
            ).hexdigest()
            
            sample = {
                'id': sample_id,
                'caption': row['caption'],
                'url': row['url'],
                'metadata': {
                    'source': f'conceptual_captions_{self.dataset}',
                    'caption_words': int(row['caption_words'])
                }
            }
            
            samples.append(sample)
        
        return samples
    
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
    
    def save_data(self, samples: List[Dict], split_name: str = None):
        """Save processed data to disk."""
        if split_name:
            # Save single split
            output_path = self.output_dir / f'{self.dataset}_{split_name}.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f)
            logger.info(f"Saved {len(samples)} samples to {output_path}")
        else:
            # Split and save all
            splits = self.split_data(samples)
            
            for split_name, split_data in splits.items():
                output_path = self.output_dir / f'{self.dataset}_{split_name}.pkl'
                with open(output_path, 'wb') as f:
                    pickle.dump(split_data, f)
                logger.info(f"Saved {len(split_data)} samples to {output_path}")
        
        # Save metadata
        metadata = {
            'dataset': self.dataset,
            'total_samples': len(samples),
            'min_caption_length': self.min_caption_length,
            'max_caption_length': self.max_caption_length,
            'validated_urls': self.validate_urls,
            'date_created': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = self.output_dir / f'{self.dataset}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def run(self):
        """Run the complete wrangling pipeline."""
        logger.info(f"Starting Conceptual Captions {self.dataset.upper()} wrangling...")
        
        urls = self.DATASET_URLS.get(self.dataset, {})
        all_samples = []
        
        for split_type, url in urls.items():
            logger.info(f"Processing {split_type} split...")
            
            # Download TSV
            tsv_path = self.download_tsv(url, f'{self.dataset}_{split_type}.tsv')
            
            # Load data
            df = self.load_tsv_data(tsv_path)
            
            # Filter samples
            df = self.filter_samples(df)
            
            # Validate URLs if requested
            if self.validate_urls:
                df = self.validate_urls_batch(df)
            
            # Process samples
            samples = self.process_samples(df)
            
            if self.dataset == 'cc3m' and split_type == 'validation':
                # CC3M has predefined validation split
                self.save_data(samples, split_name='valid')
            else:
                all_samples.extend(samples)
        
        # Save all samples (will be split automatically)
        if all_samples:
            self.save_data(all_samples)
        
        logger.info(f"Successfully processed Conceptual Captions {self.dataset.upper()}!")


def main():
    parser = argparse.ArgumentParser(description='Wrangle Conceptual Captions dataset')
    parser.add_argument('--dataset', type=str, default='cc3m',
                       choices=['cc3m', 'cc12m'],
                       help='Which dataset to download')
    parser.add_argument('--output-dir', type=str, default='data/conceptual_captions/',
                       help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to download')
    parser.add_argument('--min-caption-length', type=int, default=3,
                       help='Minimum caption length in words')
    parser.add_argument('--max-caption-length', type=int, default=256,
                       help='Maximum caption length in words')
    parser.add_argument('--validate-urls', action='store_true',
                       help='Validate image URLs (slow)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    wrangler = ConceptualCaptionsWrangler(
        dataset=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        min_caption_length=args.min_caption_length,
        max_caption_length=args.max_caption_length,
        validate_urls=args.validate_urls,
        max_workers=args.max_workers
    )
    
    wrangler.run()


if __name__ == '__main__':
    main()