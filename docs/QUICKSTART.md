# 5-Minute Quickstart

Get the CS627 Semantic-Vector Space project running in 5 minutes.

## Instant Demo

### Option 1: Automated Script

```bash
# Clone and run
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
./quickstart.sh
```

This script:
✅ Checks prerequisites  
✅ Installs dependencies  
✅ Creates demo dataset  
✅ Runs mini training  
✅ Verifies installation  

### Option 2: Docker Quick Run

```bash
# Clone repository
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627

# Pull and run
docker pull watsonwb/cs627-svs:latest
docker run --rm -it watsonwb/cs627-svs:latest python -c "
from src.Encoders import Text_to_Vec
encoder = Text_to_Vec()
result = encoder(['Hello world'])
print(f'Success! Shape: {result.shape}')
"
```

### Option 3: Make Commands

```bash
# macOS/Linux
make setup
make test

# Windows
make.bat setup
make.bat test
```

## Platform-Specific Quick Start

### macOS
```bash
# Prerequisites (2 min)
brew install python@3.10 git
brew install --cask docker

# Setup (3 min)
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
./quickstart.sh
```

### Windows
```bash
# In PowerShell/Command Prompt
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
make.bat setup
make.bat test
```

### Linux
```bash
# Prerequisites (2 min)
sudo apt install python3.10 python3-pip git

# Setup (3 min)
git clone https://github.com/WatsonWBlair/cs627.git
cd cs627
./quickstart.sh
```

## Verify Installation

```bash
# Test encoders
python -c "
from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
print('✅ Text encoder loaded')
print('✅ Audio encoder loaded')  
print('✅ Image encoder loaded')
print('All encoders working!')
"

# Check GPU (optional)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

## Quick Training Demo

```bash
# Generate demo tokens (2 min)
python -c "
import h5py, numpy as np
from pathlib import Path

# Create demo tokens
Path('data/pregenerated_tokens/demo').mkdir(parents=True, exist_ok=True)
with h5py.File('data/pregenerated_tokens/demo/train_tokens.h5', 'w') as f:
    f.create_dataset('text_base', data=np.random.randn(100, 1024))
    f.create_dataset('audio_waveform', data=np.random.randn(100, 1024))
    f.attrs['num_samples'] = 100
print('✅ Demo tokens created')
"

# Train adapter (2 min)
TOKEN_DIR=data/pregenerated_tokens/demo \
EPOCHS=3 \
BATCH_SIZE=10 \
python src/Training/train_adapters.py --mode encoder --epochs 3
```

## Next Steps

### Full Setup
```bash
# Install all dependencies
make setup

# Download real data (30 min)
make download-data

# Generate real tokens (30 min)
make tokens

# Train on real data (15 min)
make train
```

### Explore Features
```bash
# Launch Jupyter environment
make dev

# Run CLI commands
python cli.py --help
python cli.py status
python cli.py pipeline
```

### Read Documentation
- [SETUP.md](../SETUP.md) - Full installation guide
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training workflows
- [DOCKER.md](../DOCKER.md) - Container usage

## Common Issues

### "No module named X"
```bash
pip install -r requirements.txt
```

### "Docker not running"
```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker
```

### "Permission denied"
```bash
# Make scripts executable
chmod +x quickstart.sh
chmod +x docker/*.sh
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/WatsonWBlair/cs627/issues)
- **Documentation**: [Full Docs](../README.md)
- **Quick Test**: `make test`