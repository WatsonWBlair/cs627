# Data Pipeline Overview

Data acquisition, processing, and storage for the CS627 Semantic-Vector Space project.

**Technical details**: See [scripts/data_wrangling/README.md](../scripts/data_wrangling/README.md) for script APIs and usage.

**Related**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for data-related issues.

## Quick Start

```bash
# Download and extract MOSI data
python scripts/data_wrangling/download_all_mosi_videos.py
python scripts/data_wrangling/extract_all_segments.py

# Generate training triplets
python scripts/data_wrangling/wrangle_glue_data.py
python scripts/data_wrangling/wrangle_mteb_data.py
```

## Data Sources

| Dataset | Type | Size | Format | Usage |
|---------|------|------|--------|-------|
| **CMU-MOSI** | Multimodal | ~2,199 segments | .cdf, .wav, .jpg | Cross-modal training |
| **GLUE** | Text pairs | 5 tasks | Triplets | Text encoder training |
| **MTEB** | Semantic similarity | STS tasks | Pairs/triplets | Evaluation & fine-tuning |

## File Structure

```
data/
├── cmumosi/
│   ├── mosi/           # Metadata (.cdf files)
│   ├── audio/          # Extracted audio (.wav)
│   ├── frames/         # Extracted frames (.jpg)
│   └── pickles/        # Preprocessed splits (.pkl)
├── glue/
│   └── glue_triplets.pkl
└── mteb/
    └── mteb_triplets.pkl
```

## Pipeline Workflow

```
1. Download     →  2. Extract Media  →  3. Process  →  4. Split  →  5. Train
   (metadata)        (audio/frames)      (triplets)     (70/15/15)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_DOWNLOAD` | `1` | Skip SDK downloads |
| `MOSI_DATA_PATH` | `data/cmumosi/mosi/` | MOSI metadata path |
| `AUDIO_DIR` | `data/cmumosi/audio/` | Extracted audio |
| `VIDEO_DIR` | `data/cmumosi/frames/` | Extracted frames |

## Common Issues

| Issue | Solution |
|-------|----------|
| Missing .cdf files | Set `SKIP_DOWNLOAD=0` and run MOSI download |
| Audio extraction fails | Install ffmpeg: `apt install ffmpeg` |
| Video download fails | Update yt-dlp: `pip install -U yt-dlp` |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

## Documentation

- **Script usage**: [scripts/data_wrangling/README.md](../scripts/data_wrangling/README.md)
- **Token generation**: [src/Training/README.md](../src/Training/README.md)
- **Benchmarks**: [BENCHMARKS.md](BENCHMARKS.md)

## References

- [CMU-MOSI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
