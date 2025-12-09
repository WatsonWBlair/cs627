# Tech Stack

Complete overview of technologies used in the CS627 Semantic-Vector Space project.

**Installation**: See [SETUP.md](SETUP.md) for setup instructions.

**Containers**: See [DOCKER.md](DOCKER.md) for Docker configuration.

## Core ML Framework

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | >=2.0.0 | Deep learning framework |
| Transformers | >=4.30.0 | Pre-trained model access (HuggingFace) |
| Diffusers | >=0.21.0 | Diffusion-based image generation |
| Accelerate | >=0.20.0 | Distributed training, mixed precision |
| Sentence-transformers | >=2.2.0 | Pre-trained semantic embeddings |
| TRL | >=0.4.0 | Reinforcement learning training |

## Pre-trained Models

| Model | Provider | Purpose | Output Dim |
|-------|----------|---------|------------|
| `facebook/bart-base` | HuggingFace | Text encoder | 768→1024 |
| `openai/whisper-small` | HuggingFace | Audio encoder | 768→1024 |
| `microsoft/wavlm-base` | HuggingFace | Tone/emotion encoder | 768→1024 |
| `nlpconnect/vit-gpt2-image-captioning` | HuggingFace | Image encoder | 768→1024 |
| `microsoft/speecht5_tts` | HuggingFace | Audio decoder (TTS) | 1024→audio |
| `CompVis/stable-diffusion-v1-4` | HuggingFace | Image decoder | 1024→image |

## Data Processing

### Storage & Serialization

| Package | Version | Purpose |
|---------|---------|---------|
| h5py | >=2.10.0 | HDF5 format for pre-generated tokens |
| Pandas | >=1.3.0 | Data manipulation and analysis |
| NumPy | >=1.21.0 | Numerical computing |
| Datasets | >=2.10.0 | HuggingFace datasets library |

### Audio Processing

| Package | Version | Purpose |
|---------|---------|---------|
| librosa | >=0.10.0 | Audio feature extraction |
| SoundFile | >=0.12.0 | Audio file I/O |
| sentencepiece | >=0.1.96 | Tokenization for SpeechT5 |

### Image & Video Processing

| Package | Version | Purpose |
|---------|---------|---------|
| Pillow | >=9.0.0 | Image manipulation |
| OpenCV-python | >=4.5.0 | Computer vision operations |
| yt-dlp | >=2023.0.0 | Video downloading (MOSI dataset) |

## Evaluation & Benchmarks

| Package | Version | Purpose |
|---------|---------|---------|
| Evaluate | >=0.4.0 | HuggingFace evaluation metrics |
| MTEB | >=1.0.0 | Massive Text Embedding Benchmark |
| Rouge-score | >=0.1.2 | ROUGE metrics for NLP |
| NLTK | >=3.8 | Natural language toolkit |
| scikit-learn | >=1.0.0 | ML metrics and algorithms |
| SciPy | >=1.10.0 | Scientific computing |

## Infrastructure

### Containerization

| Image | Base | Size | Purpose |
|-------|------|------|---------|
| `watsonwb/cs627-svs:gpu` | pytorch:2.5.0-cuda12.4 | 8GB | GPU training |
| `watsonwb/cs627-svs:cpu` | python:3.10-slim | 4GB | CPU training |
| `watsonwb/cs627-svs:dev` | pytorch:2.5.0-cuda12.4-devel | 9GB | Jupyter development |
| `watsonwb/cs627-svs:ablation` | python:3.10-slim | 3.2GB | Ablation studies |

### Cloud Services (AWS)

| Service | Purpose |
|---------|---------|
| EC2 | GPU instances (g5.4xlarge with A10G) |
| S3 | Data storage, model weights |
| IAM | Access management |

### Cloud SDK

| Package | Version | Purpose |
|---------|---------|---------|
| boto3 | >=1.28.0 | AWS SDK for Python |

## Development Tools

### Testing

| Package | Purpose |
|---------|---------|
| pytest | Unit testing framework |
| pytest-cov | Code coverage |

### Code Quality

| Package | Purpose |
|---------|---------|
| black | Code formatter |
| flake8 | Style linter |
| mypy | Static type checker |

### Monitoring & Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| TensorBoard | - | Training visualization |
| wandb | - | Experiment tracking |
| matplotlib | >=3.5.0 | Plotting |
| seaborn | - | Statistical visualization |
| plotly | - | Interactive plots |

### Development Environment

| Package | Purpose |
|---------|---------|
| Jupyter Lab | Interactive notebooks |
| ipywidgets | Notebook widgets |
| ipdb | IPython debugger |

## CI/CD

| Tool | Purpose |
|------|---------|
| GitHub Actions | Automated testing and Docker publishing |
| Docker Buildx | Multi-platform image builds (amd64, arm64) |
| Makefile | Task automation |

## System Requirements

| Requirement | Specification |
|-------------|---------------|
| Python | 3.10 |
| CUDA | 12.4 (optional, for GPU) |
| RAM | 16GB minimum |
| GPU VRAM | 8GB+ recommended |

### System Libraries

| Library | Purpose |
|---------|---------|
| ffmpeg | Video/audio processing |
| libsndfile1 | Audio file support |
| libgomp1 | OpenMP runtime |

## External Dependencies

| Dependency | Source | Purpose |
|------------|--------|---------|
| CMU-MultimodalSDK | [GitHub](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) | MOSI/MOSEI dataset utilities |

## Data Formats

| Format | Extension | Usage |
|--------|-----------|-------|
| HDF5 | `.h5` | Pre-generated tokens |
| PyTorch | `.pth` | Model weights |
| JSON | `.json` | Metadata, configs |
| Pickle | `.pkl` | Processed datasets |
