#!/usr/bin/env python3
"""
Unified Data Preparation Pipeline

Single entry point for all data acquisition, processing, and preparation tasks.
This script coordinates all data wrangling operations for MOSI, GLUE, and MTEB datasets.

See DATA_PIPELINE.md for complete documentation.

Usage:
    # Prepare all datasets
    python scripts/prepare_data.py --datasets mosi glue mteb
    
    # Prepare specific dataset with options
    python scripts/prepare_data.py --datasets mosi --extract-media --create-splits
    
    # Full pipeline with S3 upload
    python scripts/prepare_data.py --all --upload-s3
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import shutil
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Unified data pipeline for all dataset operations."""
    
    def __init__(self, 
                 base_dir: str = "data/",
                 s3_bucket: str = "cs627-svs-data",
                 aws_region: str = "us-east-1"):
        """
        Initialize data pipeline.
        
        Args:
            base_dir: Base directory for all data
            s3_bucket: S3 bucket for cloud storage
            aws_region: AWS region for S3 operations
        """
        self.base_dir = Path(base_dir)
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region
        
        # Dataset paths
        self.mosi_path = self.base_dir / "cmumosi"
        self.glue_path = self.base_dir / "glue"
        self.mteb_path = self.base_dir / "mteb"
        
        # Ensure directories exist
        for path in [self.mosi_path, self.glue_path, self.mteb_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    # ==================== MOSI Dataset Operations ====================
    
    def download_mosi_metadata(self, force: bool = False) -> bool:
        """
        Download MOSI metadata files (.cdf format).
        
        Args:
            force: Force redownload even if files exist
            
        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info("Downloading MOSI Metadata")
        logger.info("=" * 80)
        
        mosi_data_path = self.mosi_path / "mosi"
        
        # Check if already exists
        if mosi_data_path.exists() and not force:
            cdf_files = list(mosi_data_path.glob("*.cdf"))
            if cdf_files:
                logger.info(f"Found {len(cdf_files)} .cdf files, skipping download")
                return True
        
        # Set environment to allow downloads
        os.environ['SKIP_DOWNLOAD'] = '0'
        
        try:
            from src.Training.Data_Wrangling.mosi_dataset import download_mosi
            logger.info(f"Downloading to {mosi_data_path}")
            download_mosi(str(mosi_data_path) + '/')
            
            # Verify files
            cdf_files = list(mosi_data_path.glob("*.cdf"))
            logger.info(f"Downloaded {len(cdf_files)} .cdf metadata files")
            
            # List key files
            key_files = [
                "CMU_MOSI_TimestampedWords.cdf",
                "CMU_MOSI_Opinion_Labels.cdf"
            ]
            for fname in key_files:
                fpath = mosi_data_path / fname
                if fpath.exists():
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    logger.info(f"  ✓ {fname} ({size_mb:.1f} MB)")
                else:
                    logger.warning(f"  ✗ {fname} not found")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download MOSI metadata: {e}")
            return False
        finally:
            # Reset environment
            os.environ['SKIP_DOWNLOAD'] = '1'
    
    def extract_mosi_segments(self) -> bool:
        """
        Extract audio and video segments from MOSI videos.
        
        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info("Extracting MOSI Media Segments")
        logger.info("=" * 80)
        
        audio_dir = self.mosi_path / "audio"
        frames_dir = self.mosi_path / "frames"
        
        # Check if already extracted
        audio_files = list(audio_dir.glob("*.wav"))
        frame_files = list(frames_dir.glob("*.jpg"))
        
        if audio_files or frame_files:
            logger.info(f"Found {len(audio_files)} audio and {len(frame_files)} frame files")
            logger.info("Media already extracted, skipping")
            return True
        
        # Check for extraction scripts
        extract_script = project_root / "scripts" / "data_wrangling" / "extract_all_segments.py"
        if not extract_script.exists():
            logger.warning("Extract script not found, skipping media extraction")
            logger.info("Note: Media extraction requires video files and ffmpeg")
            return False
        
        try:
            # Run extraction script
            cmd = [sys.executable, str(extract_script)]
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Media extraction completed successfully")
                return True
            else:
                logger.error(f"Media extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to extract media: {e}")
            return False
    
    def create_mosi_splits(self, 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          seed: int = 42) -> bool:
        """
        Create train/val/test splits for MOSI dataset.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            seed: Random seed for reproducibility
            
        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info("Creating MOSI Dataset Splits")
        logger.info("=" * 80)
        
        pickles_dir = self.mosi_path / "pickles"
        pickles_dir.mkdir(exist_ok=True)
        
        # Check if splits already exist
        split_files = ["preprocessed_train.pkl", "preprocessed_valid.pkl", "preprocessed_test.pkl"]
        if all((pickles_dir / f).exists() for f in split_files):
            logger.info("Splits already exist, skipping")
            return True
        
        try:
            # Use existing create_local_pickles logic
            import numpy as np
            
            # Get segment IDs from audio files or S3
            audio_dir = self.mosi_path / "audio"
            segment_ids = []
            
            if audio_dir.exists():
                for audio_file in audio_dir.glob("*.wav"):
                    # Convert: "6Egk_28TtTM_0.wav" -> "6Egk_28TtTM[0]"
                    base = audio_file.stem
                    parts = base.split("_")
                    if len(parts) >= 2:
                        video_id = "_".join(parts[:-1])
                        segment_num = parts[-1]
                        segment_id = f"{video_id}[{segment_num}]"
                        segment_ids.append(segment_id)
            
            if not segment_ids:
                logger.warning("No audio files found, trying to get IDs from S3 structure")
                # Fallback to S3-based approach from create_local_pickles.py
                try:
                    import boto3
                    s3 = boto3.client("s3", region_name=self.aws_region)
                    response = s3.list_objects_v2(
                        Bucket=self.s3_bucket, 
                        Prefix="cmumosi/audio/",
                        MaxKeys=1000
                    )
                    
                    while True:
                        if "Contents" in response:
                            for obj in response["Contents"]:
                                filename = obj["Key"].split("/")[-1]
                                if filename.endswith(".wav"):
                                    base = filename.replace(".wav", "")
                                    parts = base.split("_")
                                    if len(parts) >= 2:
                                        video_id = "_".join(parts[:-1])
                                        segment_num = parts[-1]
                                        segment_id = f"{video_id}[{segment_num}]"
                                        segment_ids.append(segment_id)
                        
                        if not response.get('IsTruncated', False):
                            break
                        response = s3.list_objects_v2(
                            Bucket=self.s3_bucket,
                            Prefix="cmumosi/audio/",
                            MaxKeys=1000,
                            ContinuationToken=response['NextContinuationToken']
                        )
                except:
                    pass
            
            if not segment_ids:
                logger.error("No segment IDs found, cannot create splits")
                return False
            
            logger.info(f"Found {len(segment_ids)} segments")
            
            # Split by video IDs
            video_ids = list(set([s.split("[")[0] for s in segment_ids]))
            logger.info(f"Found {len(video_ids)} unique videos")
            
            # Create reproducible random splits
            np.random.seed(seed)
            np.random.shuffle(video_ids)
            
            n_train = int(len(video_ids) * train_ratio)
            n_val = int(len(video_ids) * val_ratio)
            
            train_videos = set(video_ids[:n_train])
            val_videos = set(video_ids[n_train:n_train + n_val])
            test_videos = set(video_ids[n_train + n_val:])
            
            logger.info(f"Video splits: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test")
            
            # Create splits
            train_segments = [s for s in segment_ids if s.split("[")[0] in train_videos]
            val_segments = [s for s in segment_ids if s.split("[")[0] in val_videos]
            test_segments = [s for s in segment_ids if s.split("[")[0] in test_videos]
            
            logger.info(f"Segment splits: {len(train_segments)} train, {len(val_segments)} val, {len(test_segments)} test")
            
            # Save splits
            pickle_files = {
                "preprocessed_train.pkl": {"segment_ids": train_segments},
                "preprocessed_valid.pkl": {"segment_ids": val_segments},
                "preprocessed_test.pkl": {"segment_ids": test_segments}
            }
            
            for filename, data in pickle_files.items():
                filepath = pickles_dir / filename
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
                size_kb = filepath.stat().st_size / 1024
                logger.info(f"Created {filename}: {len(data['segment_ids'])} segments ({size_kb:.1f} KB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create splits: {e}")
            return False
    
    # ==================== GLUE Dataset Operations ====================
    
    def download_glue_data(self, 
                           tasks: List[str] = None,
                           max_samples: int = 50000) -> bool:
        """
        Download and process GLUE benchmark data.
        
        Args:
            tasks: List of GLUE tasks to download
            max_samples: Maximum samples per task
            
        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info("Processing GLUE Dataset")
        logger.info("=" * 80)
        
        if tasks is None:
            tasks = ['mrpc', 'qqp', 'stsb', 'mnli', 'qnli']
        
        # Check if already processed
        triplets_file = self.glue_path / "glue_triplets.pkl"
        if triplets_file.exists():
            logger.info("GLUE data already processed, skipping")
            return True
        
        try:
            # Import and run GLUE wrangler
            from scripts.data_wrangling.wrangle_glue_data import GLUEDataWrangler
            
            logger.info(f"Processing GLUE tasks: {tasks}")
            wrangler = GLUEDataWrangler(output_dir=str(self.glue_path))
            triplets, metadata = wrangler.wrangle_all_tasks(task_names=tasks)
            
            logger.info(f"Processed {len(triplets)} triplets from GLUE")
            return True
            
        except ImportError:
            logger.warning("GLUE wrangler not available, creating stub data")
            return self._create_glue_stubs(tasks)
        except Exception as e:
            logger.error(f"Failed to process GLUE data: {e}")
            return False
    
    def _create_glue_stubs(self, tasks: List[str]) -> bool:
        """Create stub GLUE data for testing."""
        logger.info("Creating GLUE stub data...")
        
        stub_triplets = [
            ("The weather is nice today.", "Today's weather is pleasant.", "The car is red."),
            ("I love this movie.", "This film is great.", "The book is boring."),
            ("She plays tennis well.", "She's good at tennis.", "He likes football."),
        ]
        
        # Save stub data
        triplets_file = self.glue_path / "glue_triplets.pkl"
        with open(triplets_file, "wb") as f:
            pickle.dump(stub_triplets, f)
        
        metadata = {
            "tasks_processed": tasks,
            "total_samples": len(stub_triplets),
            "is_stub": True,
            "created": datetime.now().isoformat()
        }
        
        metadata_file = self.glue_path / "glue_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created stub GLUE data with {len(stub_triplets)} triplets")
        return True
    
    # ==================== MTEB Dataset Operations ====================
    
    def download_mteb_data(self,
                           tasks: List[str] = None,
                           max_samples: int = 50000) -> bool:
        """
        Download and process MTEB benchmark data.
        
        Args:
            tasks: List of MTEB task types
            max_samples: Maximum samples per task
            
        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info("Processing MTEB Dataset")
        logger.info("=" * 80)
        
        if tasks is None:
            tasks = ['sts', 'retrieval']
        
        # Check if already processed
        triplets_file = self.mteb_path / "mteb_triplets.pkl"
        if triplets_file.exists():
            logger.info("MTEB data already processed, skipping")
            return True
        
        try:
            # Import and run MTEB wrangler
            from scripts.data_wrangling.wrangle_mteb_data import MTEBDataWrangler
            
            logger.info(f"Processing MTEB tasks: {tasks}")
            wrangler = MTEBDataWrangler(output_dir=str(self.mteb_path))
            triplets, metadata = wrangler.wrangle_all_tasks(task_types=tasks)
            
            logger.info(f"Processed {len(triplets)} triplets from MTEB")
            return True
            
        except ImportError:
            logger.warning("MTEB wrangler not available, creating stub data")
            return self._create_mteb_stubs(tasks)
        except Exception as e:
            logger.error(f"Failed to process MTEB data: {e}")
            return False
    
    def _create_mteb_stubs(self, tasks: List[str]) -> bool:
        """Create stub MTEB data for testing."""
        logger.info("Creating MTEB stub data...")
        
        stub_pairs = [
            ("What is machine learning?", "How does ML work?", 0.8),
            ("Python programming", "Python coding", 0.9),
            ("Deep learning models", "Neural networks", 0.85),
        ]
        
        stub_triplets = [
            ("Query about AI", "Document about artificial intelligence", "Document about cooking"),
            ("Search for Python", "Python tutorial page", "Java documentation"),
        ]
        
        # Save stub data
        pairs_file = self.mteb_path / "mteb_pairs.pkl"
        with open(pairs_file, "wb") as f:
            pickle.dump(stub_pairs, f)
        
        triplets_file = self.mteb_path / "mteb_triplets.pkl"
        with open(triplets_file, "wb") as f:
            pickle.dump(stub_triplets, f)
        
        metadata = {
            "tasks_processed": tasks,
            "total_pairs": len(stub_pairs),
            "total_triplets": len(stub_triplets),
            "is_stub": True,
            "created": datetime.now().isoformat()
        }
        
        metadata_file = self.mteb_path / "mteb_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created stub MTEB data with {len(stub_pairs)} pairs and {len(stub_triplets)} triplets")
        return True
    
    # ==================== S3 Operations ====================
    
    def upload_to_s3(self, datasets: List[str] = None) -> bool:
        """
        Upload processed data to S3.
        
        Args:
            datasets: List of datasets to upload
            
        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info("Uploading to S3")
        logger.info("=" * 80)
        
        if datasets is None:
            datasets = ['mosi', 'glue', 'mteb']
        
        try:
            import boto3
            s3 = boto3.client("s3", region_name=self.aws_region)
            
            upload_map = {
                'mosi': [
                    (self.mosi_path / "mosi", "cmumosi/mosi/"),
                    (self.mosi_path / "pickles", "cmumosi/pickles/")
                ],
                'glue': [(self.glue_path, "glue/")],
                'mteb': [(self.mteb_path, "mteb/")]
            }
            
            for dataset in datasets:
                if dataset not in upload_map:
                    continue
                
                for local_path, s3_prefix in upload_map[dataset]:
                    if not local_path.exists():
                        logger.warning(f"Path {local_path} not found, skipping")
                        continue
                    
                    logger.info(f"Uploading {local_path} to s3://{self.s3_bucket}/{s3_prefix}")
                    
                    if local_path.is_dir():
                        for file_path in local_path.rglob("*"):
                            if file_path.is_file():
                                key = s3_prefix + str(file_path.relative_to(local_path))
                                logger.debug(f"  Uploading {file_path.name} -> {key}")
                                s3.upload_file(str(file_path), self.s3_bucket, key)
                    else:
                        key = s3_prefix + local_path.name
                        s3.upload_file(str(local_path), self.s3_bucket, key)
            
            logger.info("S3 upload completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            logger.info("Make sure AWS credentials are configured: aws configure")
            return False
    
    def download_from_s3(self, datasets: List[str] = None) -> bool:
        """
        Download processed data from S3.
        
        Args:
            datasets: List of datasets to download
            
        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info("Downloading from S3")
        logger.info("=" * 80)
        
        if datasets is None:
            datasets = ['mosi', 'glue', 'mteb']
        
        try:
            import boto3
            s3 = boto3.client("s3", region_name=self.aws_region)
            
            download_map = {
                'mosi': [
                    ("cmumosi/mosi/", self.mosi_path / "mosi"),
                    ("cmumosi/pickles/", self.mosi_path / "pickles")
                ],
                'glue': [("glue/", self.glue_path)],
                'mteb': [("mteb/", self.mteb_path)]
            }
            
            for dataset in datasets:
                if dataset not in download_map:
                    continue
                
                for s3_prefix, local_path in download_map[dataset]:
                    local_path.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Downloading s3://{self.s3_bucket}/{s3_prefix} to {local_path}")
                    
                    response = s3.list_objects_v2(
                        Bucket=self.s3_bucket,
                        Prefix=s3_prefix
                    )
                    
                    if 'Contents' not in response:
                        logger.warning(f"No files found in s3://{self.s3_bucket}/{s3_prefix}")
                        continue
                    
                    for obj in response['Contents']:
                        key = obj['Key']
                        filename = key.replace(s3_prefix, '')
                        if filename:
                            file_path = local_path / filename
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            logger.debug(f"  Downloading {filename}")
                            s3.download_file(self.s3_bucket, key, str(file_path))
            
            logger.info("S3 download completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False
    
    # ==================== Validation ====================
    
    def validate_all_datasets(self) -> Dict[str, bool]:
        """
        Validate all datasets are properly prepared.
        
        Returns:
            Dictionary of dataset -> validation status
        """
        logger.info("=" * 80)
        logger.info("Validating Datasets")
        logger.info("=" * 80)
        
        results = {}
        
        # Validate MOSI
        mosi_valid = all([
            (self.mosi_path / "mosi").exists(),
            len(list((self.mosi_path / "mosi").glob("*.cdf"))) > 0,
            (self.mosi_path / "pickles" / "preprocessed_train.pkl").exists()
        ])
        results['mosi'] = mosi_valid
        logger.info(f"MOSI dataset: {'✓ Valid' if mosi_valid else '✗ Invalid'}")
        
        # Validate GLUE
        glue_valid = (self.glue_path / "glue_triplets.pkl").exists()
        results['glue'] = glue_valid
        logger.info(f"GLUE dataset: {'✓ Valid' if glue_valid else '✗ Invalid'}")
        
        # Validate MTEB
        mteb_valid = (self.mteb_path / "mteb_triplets.pkl").exists()
        results['mteb'] = mteb_valid
        logger.info(f"MTEB dataset: {'✓ Valid' if mteb_valid else '✗ Invalid'}")
        
        return results
    
    def run_full_pipeline(self, 
                         datasets: List[str] = None,
                         upload_s3: bool = False) -> bool:
        """
        Run complete data preparation pipeline.
        
        Args:
            datasets: Datasets to prepare
            upload_s3: Whether to upload to S3
            
        Returns:
            True if all steps successful
        """
        if datasets is None:
            datasets = ['mosi', 'glue', 'mteb']
        
        success = True
        
        if 'mosi' in datasets:
            success &= self.download_mosi_metadata()
            success &= self.extract_mosi_segments()
            success &= self.create_mosi_splits()
        
        if 'glue' in datasets:
            success &= self.download_glue_data()
        
        if 'mteb' in datasets:
            success &= self.download_mteb_data()
        
        if upload_s3 and success:
            success &= self.upload_to_s3(datasets)
        
        # Validate
        validation = self.validate_all_datasets()
        
        logger.info("=" * 80)
        logger.info("Pipeline Complete")
        logger.info("=" * 80)
        for dataset, valid in validation.items():
            if dataset in datasets:
                logger.info(f"{dataset}: {'✓ Ready' if valid else '✗ Failed'}")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Unified data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all datasets
  python scripts/prepare_data.py --all
  
  # Prepare specific datasets
  python scripts/prepare_data.py --datasets mosi glue
  
  # Download and process MOSI with media extraction
  python scripts/prepare_data.py --datasets mosi --download-metadata --extract-media
  
  # Upload to S3 after preparation
  python scripts/prepare_data.py --all --upload-s3
  
See DATA_PIPELINE.md for complete documentation.
        """
    )
    
    # Dataset selection
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['mosi', 'glue', 'mteb'],
        help='Datasets to prepare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Prepare all datasets (mosi, glue, mteb)'
    )
    
    # MOSI specific
    parser.add_argument(
        '--download-metadata',
        action='store_true',
        help='Download MOSI metadata files'
    )
    parser.add_argument(
        '--extract-media',
        action='store_true',
        help='Extract audio/video segments'
    )
    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Create train/val/test splits'
    )
    
    # S3 operations
    parser.add_argument(
        '--upload-s3',
        action='store_true',
        help='Upload to S3 after preparation'
    )
    parser.add_argument(
        '--download-s3',
        action='store_true',
        help='Download from S3'
    )
    parser.add_argument(
        '--s3-bucket',
        default='cs627-svs-data',
        help='S3 bucket name'
    )
    
    # Options
    parser.add_argument(
        '--base-dir',
        default='data/',
        help='Base directory for data'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only validate existing data'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = DataPipeline(
        base_dir=args.base_dir,
        s3_bucket=args.s3_bucket
    )
    
    # Determine datasets
    if args.all:
        datasets = ['mosi', 'glue', 'mteb']
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = ['mosi']  # Default to MOSI
    
    # Handle operations
    if args.validate:
        validation = pipeline.validate_all_datasets()
        success = all(validation.values())
    elif args.download_s3:
        success = pipeline.download_from_s3(datasets)
    else:
        # Run specific operations or full pipeline
        if any([args.download_metadata, args.extract_media, args.create_splits]):
            success = True
            if 'mosi' in datasets:
                if args.download_metadata:
                    success &= pipeline.download_mosi_metadata()
                if args.extract_media:
                    success &= pipeline.extract_mosi_segments()
                if args.create_splits:
                    success &= pipeline.create_mosi_splits()
        else:
            # Run full pipeline
            success = pipeline.run_full_pipeline(datasets, args.upload_s3)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()