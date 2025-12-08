# DATA_PIPELINE.md - Single Source of Truth for Data Acquisition

## Overview

This document is the **authoritative reference** for all data acquisition, processing, and storage operations in the CS627 project. All other documentation should reference this file when discussing data pipeline operations.

## Quick Start

```bash
# Prepare all datasets (MOSI, GLUE, MTEB)
python scripts/prepare_data.py --datasets mosi glue mteb

# Prepare specific dataset
python scripts/prepare_data.py --datasets mosi --split train

# Docker/EC2 workflow with S3
python scripts/prepare_data.py --datasets mosi --upload-s3
```

## Environment Variables

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `SKIP_DOWNLOAD` | `1` | Skip SDK downloads (assumes pre-staged data) | No |
| `MOSI_DATA_PATH` | `data/cmumosi/mosi/` | Path to MOSI metadata (.cdf files) | Yes |
| `AUDIO_DIR` | `data/cmumosi/audio/` | Directory for extracted audio files | Yes |
| `VIDEO_DIR` | `data/cmumosi/frames/` | Directory for extracted video frames | Yes |
| `GLUE_DATA_PATH` | `data/glue/` | Path for GLUE dataset files | No |
| `MTEB_DATA_PATH` | `data/mteb/` | Path for MTEB dataset files | No |
| `S3_BUCKET` | `cs627-svs-data` | S3 bucket for cloud storage | No |
| `AWS_REGION` | `us-east-1` | AWS region for S3 operations | No |

## Data Sources

### 1. CMU-MOSI Dataset
- **Type:** Multimodal (text, audio, video)
- **Size:** ~2,199 segments from 93 YouTube videos
- **Format:** 
  - Metadata: `.cdf` files (CMU-MultimodalSDK format)
  - Audio: `.wav` files (16kHz mono)
  - Video: `.jpg` frames (one per segment)
- **Splits:** 70% train, 15% val, 15% test

### 2. GLUE Benchmark (Text)
- **Type:** Text pairs and NLI data
- **Tasks:** MRPC, QQP, STS-B, MNLI, QNLI
- **Format:** Triplets (anchor, positive, negative)
- **Usage:** Encoder text-only training

### 3. MTEB Benchmark (Text)
- **Type:** Semantic similarity and retrieval
- **Tasks:** STS, Retrieval, Clustering, Pair Classification
- **Format:** Similarity pairs and triplets
- **Usage:** Encoder evaluation and fine-tuning

## File Structure

```
data/
├── cmumosi/
│   ├── mosi/                    # Metadata files
│   │   ├── CMU_MOSI_TimestampedWords.cdf
│   │   ├── CMU_MOSI_Opinion_Labels.cdf
│   │   └── ...
│   ├── audio/                   # Extracted audio
│   │   ├── {video_id}_{segment}.wav
│   │   └── ...
│   ├── frames/                  # Extracted frames
│   │   ├── {video_id}_{segment}.jpg
│   │   └── ...
│   └── pickles/                 # Preprocessed splits
│       ├── preprocessed_train.pkl
│       ├── preprocessed_valid.pkl
│       └── preprocessed_test.pkl
├── glue/
│   ├── glue_triplets.pkl
│   ├── glue_metadata.json
│   └── ...
└── mteb/
    ├── mteb_triplets.pkl
    ├── mteb_metadata.json
    └── ...
```

## Pipeline Steps

### Step 1: Data Download

```python
# MOSI metadata (only needed once)
from scripts.prepare_data import DataPipeline
pipeline = DataPipeline()
pipeline.download_mosi_metadata()

# GLUE data (automatic from HuggingFace)
pipeline.download_glue_data(tasks=['mrpc', 'qqp'])

# MTEB data (automatic from HuggingFace)
pipeline.download_mteb_data(tasks=['sts', 'retrieval'])
```

### Step 2: Data Processing

```python
# Extract audio/video segments from MOSI
pipeline.extract_mosi_segments()

# Process GLUE into triplets
pipeline.process_glue_triplets()

# Process MTEB into pairs/triplets
pipeline.process_mteb_pairs()
```

### Step 3: Create Splits

```python
# Create train/val/test splits
pipeline.create_splits(
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
)
```

### Step 4: S3 Upload (Optional)

```python
# Upload to S3 for distributed training
pipeline.upload_to_s3(
    bucket='cs627-svs-data',
    prefix='preprocessed/'
)
```

## Docker Workflow

### Building Image with Pre-staged Data

```bash
# 1. Prepare data locally
python scripts/prepare_data.py --datasets mosi glue mteb

# 2. Build Docker image with data
docker build -t cs627-training .

# 3. Push to registry
docker push your-registry/cs627-training:latest
```

### EC2 Instance Setup

```bash
# 1. Launch instance with setup script
./scripts/setup_remote_instance.sh

# 2. Pull Docker image
docker pull your-registry/cs627-training:latest

# 3. Run training with pre-staged data
docker run --gpus all cs627-training python src/Training/train_encoders.py
```

## S3 Bucket Structure

```
s3://cs627-svs-data/
├── cmumosi/
│   ├── mosi/               # Metadata files (.cdf)
│   ├── audio/              # Processed audio (.wav)
│   ├── frames/             # Extracted frames (.jpg)
│   └── pickles/            # Split definitions (.pkl)
├── glue/
│   └── processed/          # GLUE triplets
├── mteb/
│   └── processed/          # MTEB pairs/triplets
└── weights/
    ├── optimal/            # Best model weights
    └── checkpoints/        # Training checkpoints
```

## Adding New Datasets

### 1. Create Dataset Class

```python
# src/data/datasets/your_dataset.py
from src.data.datasets.base import BaseDataset

class YourDataset(BaseDataset):
    def download(self):
        """Download raw data"""
        
    def process(self):
        """Process into standard format"""
        
    def create_splits(self):
        """Create train/val/test splits"""
```

### 2. Register in Pipeline

```python
# scripts/prepare_data.py
DATASET_REGISTRY = {
    'mosi': MOSIDataset,
    'glue': GLUEDataset,
    'mteb': MTEBDataset,
    'your_dataset': YourDataset  # Add here
}
```

### 3. Update Documentation

Add dataset details to this file under "Data Sources" section.

## Common Issues & Solutions

### Issue: "Preprocessed data not found"
**Solution:** Run `python scripts/prepare_data.py --datasets mosi` or set `SKIP_DOWNLOAD=0`

### Issue: ".cfd file not found"
**Solution:** Files use `.cdf` extension (not `.cfd`). Check file paths.

### Issue: "Missing audio/video files"
**Solution:** Run extraction: `python scripts/prepare_data.py --extract-media`

### Issue: "S3 access denied"
**Solution:** Configure AWS credentials: `aws configure`

## Script Execution Order

For complete data preparation from scratch:

```bash
# 1. Download metadata
python scripts/prepare_data.py --download-metadata

# 2. Extract media files
python scripts/prepare_data.py --extract-media

# 3. Process datasets
python scripts/prepare_data.py --process-datasets

# 4. Create splits
python scripts/prepare_data.py --create-splits

# Or do everything at once:
python scripts/prepare_data.py --all
```

## Testing Data Pipeline

```bash
# Validate data integrity
python scripts/validate_data.py

# Test data loading
python -c "
from scripts.prepare_data import DataPipeline
pipeline = DataPipeline()
pipeline.validate_all_datasets()
"
```

## References

- [CMU-MOSI Documentation](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Project README](README.md) - General project setup
- [BENCHMARKS.md](BENCHMARKS.md) - Evaluation metrics
- [AWS_SETUP.md](AWS_SETUP.md) - Cloud configuration