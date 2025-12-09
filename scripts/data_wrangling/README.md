# Data Wrangling Scripts

Scripts for downloading, extracting, and preprocessing datasets for the CS627 Semantic-Vector Space project.

**Overview**: See [docs/DATA_PIPELINE.md](../../docs/DATA_PIPELINE.md) for data pipeline concepts and workflow.

## Architecture

All wranglers use HuggingFace datasets with streaming for memory-efficient processing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    StreamingWranglerBase                        │
│  ├── Checkpoint/resume support                                  │
│  ├── Unified triplet output format                              │
│  └── NegativePool for random sampling                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
┌───┴───┐             ┌─────┴─────┐           ┌─────┴─────┐
│ Text  │             │  Image    │           │  Audio    │
├───────┤             ├───────────┤           ├───────────┤
│ GLUE  │             │ COCO      │           │LibriSpeech│
│ MTEB  │             │ CC3M/12M  │           │ RAVDESS   │
│ CLINC │             │ LAION     │           │           │
│ MELD  │             │           │           │           │
│ MOSEI │             │           │           │           │
└───────┘             └───────────┘           └───────────┘
```

## Quick Start

```bash
# Generate triplets from any dataset
python scripts/data_wrangling/wrangle_glue_data.py
python scripts/data_wrangling/wrangle_mteb_data.py
python scripts/data_wrangling/wrangle_coco_data.py

# Multimodal emotion datasets
python scripts/data_wrangling/wrangle_mosei_data.py
python scripts/data_wrangling/wrangle_ravdess_data.py

# Combine all triplets
python scripts/data_wrangling/unified_triplet_dataset.py --stats
```

## Available Wranglers

| Script | Dataset | HuggingFace ID | Output |
|--------|---------|----------------|--------|
| `wrangle_glue_data.py` | GLUE Benchmark | `glue` | `glue_triplets.json` |
| `wrangle_mteb_data.py` | MTEB Tasks | `mteb/*`, `sentence-transformers/*` | `mteb_triplets.json` |
| `preprocess_clinc_oos.py` | CLINC-OOS | `clinc_oos` | `clinc_triplets.json` |
| `preprocess_meld.py` | MELD | `declare-lab/MELD` | `meld_triplets.json` |
| `wrangle_coco_data.py` | MS COCO | `HuggingFaceM4/COCO` | `coco_triplets.json` |
| `wrangle_librispeech_data.py` | LibriSpeech | `librispeech_asr` | `librispeech_triplets.json` |
| `wrangle_conceptual_captions.py` | CC3M/CC12M | `conceptual_captions` | `cc_triplets.json` |
| `wrangle_laion_data.py` | LAION | `laion/*` | `laion_triplets.json` |
| `wrangle_mosei_data.py` | CMU-MOSEI | `reeha-parkar/cmu-mosei-comp-seq` | `mosei_triplets.json` |
| `wrangle_ravdess_data.py` | RAVDESS | `narad/ravdess` | `ravdess_triplets.json` |

## Unified Output Format

All wranglers produce the same JSON structure:

```json
{
  "triplets": [
    {"anchor": "text1", "positive": "similar_text", "negative": "different_text"},
    ...
  ],
  "metadata": {
    "source": "dataset_name",
    "total_samples": 50000,
    "created_at": "2024-01-01T00:00:00"
  }
}
```

## Checkpoint/Resume Support

All wranglers support automatic checkpointing:

```bash
# Start processing (creates checkpoint files)
python scripts/data_wrangling/wrangle_glue_data.py

# Resume from checkpoint if interrupted
python scripts/data_wrangling/wrangle_glue_data.py

# Force fresh start
python scripts/data_wrangling/wrangle_glue_data.py --no-resume
```

Checkpoint files: `data/{dataset}/{dataset}_checkpoint.json`

## MOSI Video Pipeline

### Step 1: Download Videos

```bash
# Download all available MOSI YouTube videos
python scripts/data_wrangling/download_all_mosi_videos.py

# Retry failed downloads
python scripts/data_wrangling/download_all_mosi_videos.py --retry-failed
```

### Step 2: Extract Segments

```bash
# Extract audio + frames from all videos
python scripts/data_wrangling/extract_all_segments.py
```

### Step 3: Prepare for HuggingFace Hub

```bash
# Create Hub-ready dataset
python scripts/data_wrangling/prepare_mosi_for_hub.py
```

Output: `data/cmumosi/hub_export/` (~1.5GB)

## Unified Triplet Dataset

Combine triplets from multiple sources:

```python
from unified_triplet_dataset import UnifiedTripletDataset

# Load from directories
dataset = UnifiedTripletDataset.from_directories([
    'data/glue/',
    'data/mteb/',
    'data/clinc/',
    'data/meld/'
])

# Get PyTorch DataLoader
dataloader = dataset.to_dataloader(batch_size=32)

# Split into train/val/test
train, val, test = dataset.split()

# Save combined dataset
dataset.save('data/unified_triplets.json')
```

## Output Directory Structure

```
data/
├── glue/
│   ├── glue_triplets.json
│   └── glue_checkpoint.json
├── mteb/
│   ├── mteb_triplets.json
│   └── mteb_checkpoint.json
├── clinc/
│   └── clinc_triplets.json
├── meld/
│   └── meld_triplets.json
├── coco/
│   └── coco_triplets.json
├── librispeech/
│   └── librispeech_triplets.json
├── mosei/
│   ├── mosei_triplets.json
│   └── mosei_checkpoint.json
├── ravdess/
│   ├── ravdess_triplets.json
│   └── ravdess_checkpoint.json
├── cmumosi/
│   ├── audio/          # Extracted .wav files
│   ├── frames/         # Extracted .jpg files
│   ├── mosi/           # MOSI metadata
│   └── hub_export/     # HuggingFace Hub ready
└── unified_triplets.json
```

## Creating New Wranglers

Inherit from `StreamingWranglerBase`:

```python
from streaming_utils import StreamingWranglerBase, NegativePool

class MyWrangler(StreamingWranglerBase):
    def __init__(self, output_dir: str = "data/mydataset/"):
        super().__init__("mydataset", output_dir)
        self.negative_pool = NegativePool(max_size=10000)

    def process(self) -> List[Dict[str, str]]:
        dataset = load_dataset("my/dataset", streaming=True)

        for sample in dataset:
            # Add to negative pool
            self.negative_pool.add(sample['text'])

            # Generate triplet
            self.triplets.append({
                'anchor': sample['anchor'],
                'positive': sample['positive'],
                'negative': self.negative_pool.sample()
            })

        return self.triplets

# Run with checkpoint support
wrangler = MyWrangler()
wrangler.run(resume=True)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_INTERVAL` | `5000` | Samples between checkpoints |

## Troubleshooting

### "HuggingFace dataset loading error"

```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets
python scripts/data_wrangling/wrangle_glue_data.py
```

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

## Related Documentation

- [docs/DATA_PIPELINE.md](../../docs/DATA_PIPELINE.md) - Pipeline overview
- [docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) - General troubleshooting
- [src/Training/Data_Wrangling/](../../src/Training/Data_Wrangling/) - Runtime data loading
