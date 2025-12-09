"""
MS COCO Dataset Wrangler for Image-Text Training

This script downloads and processes the MS COCO (Common Objects in Context) dataset
for training multimodal encoders. COCO is a large-scale object detection, segmentation,
and captioning dataset with 120K+ images and 5 captions per image.

Features:
- Download COCO images and annotations
- Extract multiple captions per image
- Create training triplets for contrastive learning
- Support for Karpathy splits (standard for vision-language)
- Image preprocessing options

Usage:
    # Download COCO 2017 dataset
    python scripts/data_wrangling/wrangle_coco_data.py --year 2017
    
    # Download with Karpathy splits
    python scripts/data_wrangling/wrangle_coco_data.py --use-karpathy-split
    
    # Download subset for testing
    python scripts/data_wrangling/wrangle_coco_data.py --max-images 10000
"""

import os
import sys
import json
import pickle
import argparse
import hashlib
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from collections import defaultdict
import zipfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import COCO API
try:
    from pycocotools.coco import COCO
    COCO_API_AVAILABLE = True
except ImportError:
    COCO_API_AVAILABLE = False
    logger.warning("pycocotools not installed. Install with: pip install pycocotools")


class COCODataWrangler:
    """Download and process MS COCO dataset for training."""
    
    # COCO dataset URLs
    COCO_URLS = {
        '2014': {
            'train_images': 'http://images.cocodataset.org/zips/train2014.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2014.zip',
            'train_val_annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
        },
        '2017': {
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
    }
    
    # Karpathy split JSON (standard for vision-language tasks)
    KARPATHY_SPLIT_URL = 'https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip'
    
    def __init__(self,
                 output_dir: str = 'data/coco/',
                 year: str = '2017',
                 download_images: bool = False,
                 use_karpathy_split: bool = True,
                 max_images: Optional[int] = None,
                 min_caption_length: int = 5,
                 max_caption_length: int = 100):
        """
        Initialize COCO data wrangler.
        
        Args:
            output_dir: Directory to save processed data
            year: COCO dataset year ('2014' or '2017')
            download_images: Whether to download actual images (large!)
            use_karpathy_split: Use standard Karpathy train/val/test splits
            max_images: Maximum number of images to process
            min_caption_length: Minimum caption length in words
            max_caption_length: Maximum caption length in words
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.year = year
        self.download_images = download_images
        self.use_karpathy_split = use_karpathy_split
        self.max_images = max_images
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length
        
        # Paths
        self.annotations_dir = self.output_dir / 'annotations'
        self.images_dir = self.output_dir / f'images{year}'
        self.cache_dir = self.output_dir / 'cache'
        
        for dir_path in [self.annotations_dir, self.images_dir, self.cache_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def download_file(self, url: str, dest_path: Path, desc: str = None) -> Path:
        """Download file with progress bar."""
        if dest_path.exists():
            logger.info(f"File already exists: {dest_path}")
            return dest_path
        
        logger.info(f"Downloading {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded to {dest_path}")
        return dest_path
    
    def extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract zip file."""
        logger.info(f"Extracting {zip_path}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.namelist(), desc="Extracting"):
                zip_ref.extract(member, extract_to)
        
        logger.info(f"Extracted to {extract_to}")
    
    def download_annotations(self):
        """Download COCO annotations."""
        urls = self.COCO_URLS[self.year]
        
        if 'annotations' in urls:
            anno_url = urls['annotations']
        else:
            anno_url = urls['train_val_annotations']
        
        # Download annotations zip
        anno_zip = self.cache_dir / f'annotations_{self.year}.zip'
        self.download_file(anno_url, anno_zip, f"COCO {self.year} annotations")
        
        # Extract if not already extracted
        anno_file = self.annotations_dir / f'captions_train{self.year}.json'
        if not anno_file.exists():
            self.extract_zip(anno_zip, self.output_dir)
    
    def download_karpathy_split(self):
        """Download Karpathy split annotations."""
        karpathy_zip = self.cache_dir / 'caption_datasets.zip'
        self.download_file(self.KARPATHY_SPLIT_URL, karpathy_zip, "Karpathy splits")
        
        # Extract if needed
        karpathy_json = self.annotations_dir / 'dataset_coco.json'
        if not karpathy_json.exists():
            self.extract_zip(karpathy_zip, self.annotations_dir)
            
            # Move the COCO json to the right place
            extracted_path = self.annotations_dir / 'dataset_coco.json'
            if not extracted_path.exists():
                # Sometimes it's in a subdirectory
                for json_file in self.annotations_dir.rglob('dataset_coco.json'):
                    shutil.move(str(json_file), str(extracted_path))
                    break
    
    def load_annotations(self) -> Dict:
        """Load COCO caption annotations."""
        if self.use_karpathy_split:
            # Load Karpathy split
            json_path = self.annotations_dir / 'dataset_coco.json'
            if not json_path.exists():
                self.download_karpathy_split()
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded Karpathy split with {len(data['images'])} images")
            return data
        
        else:
            # Load official COCO annotations
            if not COCO_API_AVAILABLE:
                raise ImportError("pycocotools required for official annotations")
            
            # Download if needed
            train_anno = self.annotations_dir / f'captions_train{self.year}.json'
            val_anno = self.annotations_dir / f'captions_val{self.year}.json'
            
            if not train_anno.exists():
                self.download_annotations()
            
            # Load with COCO API
            data = {'train': [], 'val': []}
            
            for split, anno_file in [('train', train_anno), ('val', val_anno)]:
                coco = COCO(anno_file)
                img_ids = coco.getImgIds()
                
                for img_id in tqdm(img_ids, desc=f"Loading {split}"):
                    img_info = coco.loadImgs(img_id)[0]
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    anns = coco.loadAnns(ann_ids)
                    
                    captions = [ann['caption'] for ann in anns]
                    
                    data[split].append({
                        'filename': img_info['file_name'],
                        'imgid': img_id,
                        'captions': captions,
                        'split': split
                    })
            
            return data
    
    def process_karpathy_data(self, data: Dict) -> Dict[str, List[Dict]]:
        """Process Karpathy split data into our format."""
        splits = defaultdict(list)
        
        for item in tqdm(data['images'], desc="Processing images"):
            split = item['split']
            if split == 'restval':
                split = 'train'  # Merge restval into train
            
            # Skip if we're limiting images
            if self.max_images and len(splits[split]) >= self.max_images:
                continue
            
            # Extract captions
            captions = []
            for sentence in item.get('sentences', []):
                caption = sentence.get('raw', sentence.get('tokens', ''))
                if isinstance(caption, list):
                    caption = ' '.join(caption)
                
                # Filter by length
                words = caption.split()
                if self.min_caption_length <= len(words) <= self.max_caption_length:
                    captions.append(caption)
            
            if not captions:
                continue
            
            # Create sample
            sample = {
                'id': item.get('cocoid', item.get('imgid')),
                'filename': item['filename'],
                'captions': captions,
                'split': split
            }
            
            # Add image path if available
            if 'filepath' in item:
                sample['filepath'] = item['filepath']
            
            splits[split].append(sample)
        
        # Rename 'val' to 'valid' for consistency
        if 'val' in splits:
            splits['valid'] = splits.pop('val')
        
        logger.info(f"Processed splits: {', '.join(f'{k}: {len(v)}' for k, v in splits.items())}")
        
        return dict(splits)
    
    def process_official_data(self, data: Dict) -> Dict[str, List[Dict]]:
        """Process official COCO data into our format."""
        splits = {}
        
        for split_name in ['train', 'val']:
            if split_name not in data:
                continue
            
            samples = []
            for item in tqdm(data[split_name], desc=f"Processing {split_name}"):
                # Filter captions by length
                captions = []
                for caption in item['captions']:
                    words = caption.split()
                    if self.min_caption_length <= len(words) <= self.max_caption_length:
                        captions.append(caption)
                
                if not captions:
                    continue
                
                sample = {
                    'id': item['imgid'],
                    'filename': item['filename'],
                    'captions': captions
                }
                
                samples.append(sample)
                
                if self.max_images and len(samples) >= self.max_images:
                    break
            
            # Rename val to valid
            if split_name == 'val':
                splits['valid'] = samples
            else:
                splits[split_name] = samples
        
        # Create test split from validation if not present
        if 'test' not in splits and 'valid' in splits:
            n_valid = len(splits['valid'])
            n_test = n_valid // 2
            splits['test'] = splits['valid'][:n_test]
            splits['valid'] = splits['valid'][n_test:]
        
        logger.info(f"Processed splits: {', '.join(f'{k}: {len(v)}' for k, v in splits.items())}")
        
        return splits
    
    def create_training_pairs(self, samples: List[Dict]) -> List[Dict]:
        """Create training pairs from image-caption data."""
        training_data = []
        
        for sample in tqdm(samples, desc="Creating training pairs"):
            img_id = sample['id']
            captions = sample['captions']
            
            # Create positive pairs (each caption with the image)
            for i, caption in enumerate(captions):
                pair = {
                    'id': f"{img_id}_{i}",
                    'image_id': img_id,
                    'image_file': sample['filename'],
                    'caption': caption,
                    'metadata': {
                        'source': 'coco',
                        'year': self.year,
                        'num_captions': len(captions)
                    }
                }
                
                # Add other captions as additional positives
                pair['other_captions'] = [c for j, c in enumerate(captions) if j != i]
                
                training_data.append(pair)
        
        return training_data
    
    def save_data(self, splits: Dict[str, List[Dict]]):
        """Save processed data to disk."""
        for split_name, samples in splits.items():
            # Create training pairs
            training_data = self.create_training_pairs(samples)
            
            # Save pickle file
            output_path = self.output_dir / f'coco_{split_name}.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(training_data, f)
            logger.info(f"Saved {len(training_data)} pairs to {output_path}")
            
            # Save metadata
            metadata_path = self.output_dir / f'coco_{split_name}_metadata.json'
            metadata = {
                'num_images': len(samples),
                'num_pairs': len(training_data),
                'avg_captions_per_image': np.mean([len(s['captions']) for s in samples]),
                'year': self.year,
                'use_karpathy': self.use_karpathy_split
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save overall metadata
        overall_metadata = {
            'dataset': 'coco',
            'year': self.year,
            'splits': list(splits.keys()),
            'total_images': sum(len(s) for s in splits.values()),
            'use_karpathy_split': self.use_karpathy_split,
            'min_caption_length': self.min_caption_length,
            'max_caption_length': self.max_caption_length,
            'download_images': self.download_images
        }
        
        metadata_path = self.output_dir / 'coco_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(overall_metadata, f, indent=2)
        logger.info(f"Saved overall metadata to {metadata_path}")
    
    def download_images_for_splits(self, splits: Dict[str, List[Dict]]):
        """Download actual image files (optional, very large!)."""
        if not self.download_images:
            logger.info("Skipping image download (use --download-images to enable)")
            return
        
        logger.warning("Downloading COCO images requires ~20GB of disk space!")
        
        urls = self.COCO_URLS[self.year]
        
        # Download train images
        if 'train' in splits:
            train_zip = self.cache_dir / f'train{self.year}.zip'
            self.download_file(urls['train_images'], train_zip, f"COCO train images")
            
            train_dir = self.images_dir / f'train{self.year}'
            if not train_dir.exists():
                self.extract_zip(train_zip, self.images_dir)
        
        # Download val images
        if 'valid' in splits or 'test' in splits:
            val_zip = self.cache_dir / f'val{self.year}.zip'
            self.download_file(urls['val_images'], val_zip, f"COCO val images")
            
            val_dir = self.images_dir / f'val{self.year}'
            if not val_dir.exists():
                self.extract_zip(val_zip, self.images_dir)
    
    def run(self):
        """Run the complete wrangling pipeline."""
        logger.info(f"Starting COCO {self.year} data wrangling...")
        
        # Load annotations
        data = self.load_annotations()
        
        # Process into our format
        if self.use_karpathy_split:
            splits = self.process_karpathy_data(data)
        else:
            splits = self.process_official_data(data)
        
        # Save processed data
        self.save_data(splits)
        
        # Optionally download images
        self.download_images_for_splits(splits)
        
        logger.info(f"Successfully processed COCO {self.year} dataset!")


def main():
    parser = argparse.ArgumentParser(description='Wrangle MS COCO dataset')
    parser.add_argument('--year', type=str, default='2017',
                       choices=['2014', '2017'],
                       help='COCO dataset year')
    parser.add_argument('--output-dir', type=str, default='data/coco/',
                       help='Output directory')
    parser.add_argument('--download-images', action='store_true',
                       help='Download actual images (20GB+)')
    parser.add_argument('--use-karpathy-split', action='store_true', default=True,
                       help='Use standard Karpathy splits')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--min-caption-length', type=int, default=5,
                       help='Minimum caption length in words')
    parser.add_argument('--max-caption-length', type=int, default=100,
                       help='Maximum caption length in words')
    
    args = parser.parse_args()
    
    wrangler = COCODataWrangler(
        output_dir=args.output_dir,
        year=args.year,
        download_images=args.download_images,
        use_karpathy_split=args.use_karpathy_split,
        max_images=args.max_images,
        min_caption_length=args.min_caption_length,
        max_caption_length=args.max_caption_length
    )
    
    wrangler.run()


if __name__ == '__main__':
    main()