# Scripts Directory

This directory contains production-ready scripts for the CS627 Semantic-Vector Space project.

## Directory Structure

```
scripts/
├── run_evaluation.py              # Main evaluation runner
├── run_full_evaluation.py         # Comprehensive evaluation with visualizations
├── generate_performance_figures.py # Generate publication-quality figures
├── setup_remote_instance.sh      # Cloud instance setup script
├── cloud/                        # Cloud deployment utilities
│   ├── README.md                 # Cloud deployment guide
│   ├── download_from_cloud.sh   # Download results from cloud
│   └── setup_and_train.sh       # Automated cloud training
└── data_wrangling/              # Data preparation scripts
    ├── download_test_videos.py  # Download MOSI test videos
    ├── download_all_mosi_videos.py # Download all MOSI videos
    ├── extract_test_segments.py # Extract audio/frames from test videos
    ├── extract_all_segments.py  # Extract all segments
    ├── wrangle_mteb_data.py    # Prepare MTEB benchmark data
    ├── wrangle_glue_data.py    # Prepare GLUE benchmark data
    ├── preprocess_clinc_oos.py # Preprocess intent classification data
    ├── preprocess_meld.py       # Preprocess emotion classification data
    └── utils/                    # Extraction utilities
        ├── mosi_audio_extractor.py # Audio extraction helper
        └── mosi_frame_extractor.py # Video frame extraction helper
```

## Quick Start

### 1. Data Preparation
```bash
# Download and prepare MOSI data
python scripts/data_wrangling/download_test_videos.py
python scripts/data_wrangling/extract_test_segments.py

# Prepare benchmark data (optional)
python scripts/data_wrangling/wrangle_mteb_data.py
python scripts/data_wrangling/wrangle_glue_data.py
```

### 2. Evaluation
```bash
# Quick evaluation
python scripts/run_evaluation.py

# Full evaluation with all metrics
python scripts/run_full_evaluation.py

# Generate figures for papers
python scripts/generate_performance_figures.py --publication --dpi 300
```

### 3. Cloud Deployment
```bash
# Setup remote instance
bash scripts/setup_remote_instance.sh user@remote-host

# Or use cloud utilities
bash scripts/cloud/setup_and_train.sh
```

## Script Descriptions

### Evaluation Scripts

#### `run_evaluation.py`
Basic evaluation script for cross-modal retrieval and alignment metrics.
- Calculates Recall@K for all modality pairs
- Computes alignment quality metrics
- Outputs results to console and JSON

#### `run_full_evaluation.py`
Comprehensive evaluation with all metrics and visualizations.
- Runs all evaluation metrics
- Generates t-SNE/UMAP visualizations
- Creates HTML dashboard report
- Evaluates on MTEB/GLUE benchmarks (if available)

#### `generate_performance_figures.py`
Generates publication-quality figures from trained models.
- Cross-modal retrieval confusion matrices
- Semantic space visualizations
- Training convergence curves
- Benchmark comparison charts

### Data Wrangling Scripts

#### MOSI Data Scripts
- `download_test_videos.py`: Downloads 5-10 test videos for quick testing
- `download_all_mosi_videos.py`: Downloads complete MOSI dataset
- `extract_test_segments.py`: Extracts audio and frames from test videos
- `extract_all_segments.py`: Processes entire dataset

#### Benchmark Data Scripts
- `wrangle_mteb_data.py`: Extracts training triplets from MTEB tasks
- `wrangle_glue_data.py`: Extracts training triplets from GLUE tasks
- `preprocess_clinc_oos.py`: Prepares intent classification data
- `preprocess_meld.py`: Prepares emotion classification data

### Cloud Scripts

#### `setup_remote_instance.sh`
Automated setup for cloud GPU instances (AWS, GCP, Azure, Lambda Labs).
- Detects and installs GPU drivers
- Sets up Python environment
- Clones repository and installs dependencies
- Configures for immediate training

#### `cloud/` Directory
Contains additional cloud deployment utilities and documentation.

## Recent Cleanup

A comprehensive cleanup was performed on December 4, 2024:
- Removed 5 redundant test scripts
- Moved 4 validation scripts to `tests/validation/`
- Organized all scripts by function
- Improved documentation and naming consistency

## Notes

- All scripts include comprehensive docstrings and usage examples
- Test and validation scripts have been moved to `tests/` directory
- Production scripts only - no development/debug scripts
- Follow naming convention: `{action}_{target}.py`

## Related Documentation

- [Training Guide](../src/Training/README.md)
- [Evaluation Guide](../src/Evaluation/README.md)
- [Cloud Deployment](cloud/README.md)
- [Data Preparation](../src/Training/Data_Wrangling/README.md)