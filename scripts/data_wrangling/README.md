# Data Wrangling Scripts

Scripts for downloading, extracting, and preprocessing datasets for the CS627 Semantic-Vector Space project.

**Overview**: See [docs/DATA_PIPELINE.md](../../docs/DATA_PIPELINE.md) for data pipeline concepts and workflow.

## Available Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `download_all_mosi_videos.py` | Download MOSI YouTube videos | Video IDs | `.mp4` files |
| `download_test_videos.py` | Download subset for testing | Video IDs | `.mp4` files |
| `extract_all_segments.py` | Extract audio/frames from videos | `.mp4` files | `.wav`, `.jpg` |
| `extract_test_segments.py` | Extract subset for testing | `.mp4` files | `.wav`, `.jpg` |
| `wrangle_glue_data.py` | Process GLUE into triplets | HuggingFace | `glue_triplets.pkl` |
| `wrangle_mteb_data.py` | Process MTEB into pairs | HuggingFace | `mteb_triplets.pkl` |
| `preprocess_meld.py` | Process MELD dataset | MELD files | Processed `.pkl` |
| `preprocess_clinc_oos.py` | Process CLINC-OOS dataset | CLINC files | Processed `.pkl` |

## Quick Start

```bash
# Download and extract MOSI data
python scripts/data_wrangling/download_all_mosi_videos.py
python scripts/data_wrangling/extract_all_segments.py

# Generate training triplets from GLUE/MTEB
python scripts/data_wrangling/wrangle_glue_data.py
python scripts/data_wrangling/wrangle_mteb_data.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOSI_DATA_PATH` | `data/cmumosi/mosi/` | MOSI metadata directory |
| `AUDIO_DIR` | `data/cmumosi/audio/` | Output for extracted audio |
| `VIDEO_DIR` | `data/cmumosi/frames/` | Output for extracted frames |
| `GLUE_DATA_PATH` | `data/glue/` | GLUE output directory |
| `MTEB_DATA_PATH` | `data/mteb/` | MTEB output directory |
| `SKIP_DOWNLOAD` | `1` | Skip SDK downloads |

## MOSI Video Pipeline

### Step 1: Download Videos

```bash
# Full dataset (~93 videos)
python scripts/data_wrangling/download_all_mosi_videos.py

# Test subset (~10 videos)
python scripts/data_wrangling/download_test_videos.py
```

**Requirements**: `yt-dlp` or `youtube-dl` installed.

### Step 2: Extract Segments

```bash
# Extract all segments based on MOSI metadata timestamps
python scripts/data_wrangling/extract_all_segments.py

# Test subset only
python scripts/data_wrangling/extract_test_segments.py
```

**Output**:
- `data/cmumosi/audio/{video_id}_{segment}.wav` - 16kHz mono audio
- `data/cmumosi/frames/{video_id}_{segment}.jpg` - Representative frame

### Segment Extraction Details

```python
# Audio extraction settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1  # Mono
AUDIO_FORMAT = 'wav'

# Frame extraction settings
FRAME_FORMAT = 'jpg'
FRAME_QUALITY = 95
```

## GLUE Data Wrangling

### Generate Triplets

```bash
python scripts/data_wrangling/wrangle_glue_data.py
```

**Tasks processed**:
- MRPC (paraphrase detection)
- QQP (question duplicates)
- STS-B (semantic similarity)
- MNLI (natural language inference)
- QNLI (question-answering NLI)

**Output format** (`data/glue/glue_triplets.pkl`):
```python
{
    'triplets': [
        {'anchor': str, 'positive': str, 'negative': str},
        ...
    ],
    'metadata': {
        'source_tasks': ['mrpc', 'qqp', ...],
        'total_triplets': int,
        'created_at': str
    }
}
```

### Triplet Generation Logic

```python
# From paraphrase pairs (MRPC, QQP)
# Positive: paraphrase pair
# Negative: random non-paraphrase

# From NLI data (MNLI, QNLI)
# Positive: entailment pair
# Negative: contradiction pair

# From similarity scores (STS-B)
# Positive: high similarity (>4.0)
# Negative: low similarity (<2.0)
```

## MTEB Data Wrangling

### Generate Pairs

```bash
python scripts/data_wrangling/wrangle_mteb_data.py
```

**Tasks processed**:
- STS12-16 (semantic similarity)
- STSBenchmark
- SICK-R (semantic relatedness)

**Output format** (`data/mteb/mteb_triplets.pkl`):
```python
{
    'pairs': [
        {'text1': str, 'text2': str, 'score': float},
        ...
    ],
    'triplets': [
        {'anchor': str, 'positive': str, 'negative': str},
        ...
    ],
    'metadata': {...}
}
```

## Output File Structure

```
data/
├── cmumosi/
│   ├── mosi/                    # Metadata (.cdf files)
│   ├── audio/                   # Extracted audio
│   │   └── {video_id}_{segment}.wav
│   ├── frames/                  # Extracted frames
│   │   └── {video_id}_{segment}.jpg
│   └── pickles/                 # Preprocessed splits
│       ├── preprocessed_train.pkl
│       ├── preprocessed_valid.pkl
│       └── preprocessed_test.pkl
├── glue/
│   ├── glue_triplets.pkl
│   └── glue_metadata.json
└── mteb/
    ├── mteb_triplets.pkl
    └── mteb_metadata.json
```

## Adding New Dataset Scripts

### Template

```python
#!/usr/bin/env python3
"""
{Dataset} data wrangling script.

Usage:
    python scripts/data_wrangling/wrangle_{dataset}_data.py

Output:
    data/{dataset}/{dataset}_triplets.pkl
"""

import os
import pickle
from datasets import load_dataset

OUTPUT_DIR = os.environ.get('{DATASET}_DATA_PATH', 'data/{dataset}/')

def download_data():
    """Download raw data from source."""
    pass

def process_to_triplets(data):
    """Convert raw data to anchor/positive/negative triplets."""
    pass

def save_output(triplets, metadata):
    """Save processed data to pickle file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {'triplets': triplets, 'metadata': metadata}
    with open(os.path.join(OUTPUT_DIR, '{dataset}_triplets.pkl'), 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    data = download_data()
    triplets = process_to_triplets(data)
    save_output(triplets, {'source': '{dataset}', 'count': len(triplets)})
```

## Troubleshooting

### "Video download failed"
```bash
# Update yt-dlp
pip install -U yt-dlp

# Check video availability
yt-dlp --list-formats "VIDEO_URL"
```

### "Audio extraction failed"
```bash
# Install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### "Missing .cdf metadata files"
```bash
# Download MOSI metadata
SKIP_DOWNLOAD=0 python -c "
from src.Training.Data_Wrangling.mosi_dataset import download_mosi
download_mosi('data/cmumosi/mosi/')
"
```

### "HuggingFace dataset loading error"
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets
python scripts/data_wrangling/wrangle_glue_data.py
```

## Related Documentation

- [docs/DATA_PIPELINE.md](../../docs/DATA_PIPELINE.md) - Pipeline overview and workflow
- [docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) - General troubleshooting
- [src/Training/Data_Wrangling/](../../src/Training/Data_Wrangling/) - Runtime data loading
