# CS627 Troubleshooting Guide

Common issues and solutions for the Semantic-Vector Space project.

## Installation Issues

### Python Dependencies

**Problem**: Package conflicts or version mismatches
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solution**:
```bash
# Create fresh virtual environment
python -m venv venv_fresh
source venv_fresh/bin/activate  # Linux/Mac
# or
venv_fresh\Scripts\activate  # Windows

# Install with specific versions
pip install -r requirements.txt --no-cache-dir

# If still failing, install core packages first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
```

### CUDA/GPU Issues

**Problem**: CUDA not available
```python
RuntimeError: CUDA is not available
```

**Solution**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
nvidia-smi

# Run on CPU if no GPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

## Data Issues

### MOSI Dataset

**Problem**: MOSI data not downloading
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/cmumosi/mosi/'
```

**Solution**:
```bash
# Force download
SKIP_DOWNLOAD=0 python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"

# Manual download if SDK fails
mkdir -p data/cmumosi/mosi
cd data/cmumosi
wget http://multicomp.cs.cmu.edu/raw_datasets/CMU_MOSI.zip
unzip CMU_MOSI.zip
```

**Problem**: Audio/video files missing
```
FileNotFoundError: Audio file not found
```

**Solution**:
```bash
# Extract audio from videos
python scripts/extract_audio.py --input data/cmumosi/videos --output data/cmumosi/audio

# Extract frames from videos
python scripts/extract_frames.py --input data/cmumosi/videos --output data/cmumosi/frames
```

### Token Generation

**Problem**: No tokens found
```
ValueError: No token files found in data/pregenerated_tokens/mosi/
```

**Solution**:
```bash
# Generate tokens
python src/Training/pregenerate_tokens.py

# Verify token files
ls -lh data/pregenerated_tokens/mosi/
# Should see: train_tokens.h5, val_tokens.h5, test_tokens.h5
```

**Problem**: Token generation OOM
```
RuntimeError: CUDA out of memory during token generation
```

**Solution**:
```bash
# Reduce batch size for generation
BATCH_SIZE=8 python src/Training/pregenerate_tokens.py

# Use CPU for generation (slower)
CUDA_VISIBLE_DEVICES="" python src/Training/pregenerate_tokens.py

# Process in chunks
python src/Training/pregenerate_tokens.py --chunk-size 100
```

## Training Issues

### Memory Problems

**Problem**: CUDA out of memory during training
```
RuntimeError: CUDA out of memory. Tried to allocate X MB
```

**Solutions**:
```bash
# Solution 1: Reduce batch size
BATCH_SIZE=64 python src/Training/train_adapters.py

# Solution 2: Use gradient accumulation
GRADIENT_ACCUMULATION=4 BATCH_SIZE=32 python src/Training/train_adapters.py

# Solution 3: Enable mixed precision
USE_AMP=1 python src/Training/train_adapters.py

# Solution 4: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Solution 5: Monitor GPU memory
watch -n 1 nvidia-smi
```

### Convergence Problems

**Problem**: Loss not decreasing
```
Epoch 10: Loss still at 2.0+
```

**Solutions**:
```bash
# Lower learning rate
python src/Training/train_adapters.py --lr 0.0001

# Check data normalization
python scripts/check_token_stats.py

# Increase model capacity
ADAPTER_HIDDEN_SIZE=1024 ADAPTER_LAYERS=3 python src/Training/train_adapters.py

# Use different optimizer
python src/Training/train_adapters.py --optimizer adamw
```

**Problem**: Validation metrics plateauing
```
Val recall@1 stuck at 30%
```

**Solutions**:
```bash
# Add regularization
python src/Training/train_adapters.py --dropout 0.2 --weight-decay 0.01

# Use learning rate scheduling
python src/Training/train_adapters.py --scheduler cosine --warmup-epochs 5

# Early stopping
python src/Training/train_adapters.py --early-stopping-patience 10

# Data augmentation
python src/Training/train_adapters.py --augment-tokens
```

### NaN/Inf Losses

**Problem**: Loss becomes NaN or Inf
```
Loss: nan, Recall@1: 0.00
```

**Solutions**:
```bash
# Gradient clipping
python src/Training/train_adapters.py --gradient-clip 1.0

# Lower learning rate
python src/Training/train_adapters.py --lr 0.00001

# Check for bad data
python scripts/validate_tokens.py

# Disable mixed precision
USE_AMP=0 python src/Training/train_adapters.py
```

## Model Loading Issues

### Weight Mismatches

**Problem**: Size mismatch when loading adapter weights
```
RuntimeError: Error(s) in loading state_dict: size mismatch
```

**Solution**:
```python
# Check adapter configuration
import torch
weights = torch.load('OptimalWeights/text_adapter_weights.pth')
print(weights.keys())
print({k: v.shape for k, v in weights.items()})

# Load with strict=False
adapter.load_state_dict(weights, strict=False)

# Recreate adapter with matching config
from src.utils.Adapter import Adapter
adapter = Adapter(hidden_size=512, num_layers=2)  # Match saved config
```

### Missing Weights

**Problem**: Adapter weights not found
```
FileNotFoundError: OptimalWeights/text_adapter_weights.pth not found
```

**Solution**:
```bash
# Train adapters first
python src/Training/train_adapters.py --mode encoder

# Download pre-trained weights
wget https://example.com/pretrained/text_adapter_weights.pth -P OptimalWeights/

# Check weights directory
ls -la OptimalWeights/
```

## Inference Issues

### Encoder/Decoder Errors

**Problem**: Model not producing expected output dimensions
```
ValueError: Expected output shape (batch, 1024), got (batch, 768)
```

**Solution**:
```python
# Verify encoder output dimensions
from Encoders import Text_to_Vec
encoder = Text_to_Vec()
output = encoder(["test text"])
print(output.shape)  # Should be (1, 1024)

# Check adapter configuration
from src.utils.Adapter import Adapter
adapter = Adapter(input_dim=1024, output_dim=1024)
```

### Performance Issues

**Problem**: Inference too slow
```
Processing 1 sample takes >5 seconds
```

**Solutions**:
```python
# Use batch processing
texts = ["text1", "text2", "text3"]
outputs = encoder(texts)  # Process all at once

# Enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)

# Use mixed precision
with torch.cuda.amp.autocast():
    outputs = encoder(texts)

# Cache models
encoder = encoder.eval()  # Disable dropout
with torch.no_grad():  # Disable gradient computation
    outputs = encoder(texts)
```

## Docker Issues

### Build Failures

**Problem**: Docker build fails
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt"
```

**Solution**:
```bash
# Clear Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t cs627-svs .

# Use pre-built image
docker pull watsonwb/cs627-svs:latest
```

### Container GPU Access

**Problem**: GPU not accessible in container
```
RuntimeError: No CUDA GPUs are available
```

**Solution**:
```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with GPU support
docker run --gpus all cs627-svs

# Verify GPU in container
docker run --gpus all cs627-svs nvidia-smi
```

## AWS/Cloud Issues

### Instance Setup

**Problem**: UserData script not running
```
Tokens not generated after instance launch
```

**Solution**:
```bash
# Check cloud-init logs
sudo cat /var/log/cloud-init-output.log

# Run setup manually
cd /home/ubuntu/cs627
source venv/bin/activate
python src/Training/pregenerate_tokens.py

# Verify S3 permissions
aws s3 ls s3://your-bucket/
```

### S3 Sync Issues

**Problem**: Can't upload results to S3
```
Access Denied (Service: Amazon S3; Status Code: 403)
```

**Solution**:
```bash
# Check IAM role
aws sts get-caller-identity

# Verify bucket permissions
aws s3api get-bucket-acl --bucket your-bucket

# Use correct region
export AWS_DEFAULT_REGION=us-east-1

# Sync with proper flags
aws s3 sync OptimalWeights/ s3://your-bucket/weights/ --storage-class REDUCED_REDUNDANCY
```

## Environment Issues

### Path Problems

**Problem**: Module not found errors
```
ModuleNotFoundError: No module named 'src'
```

**Solution**:
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in Python
import sys
sys.path.append('/path/to/cs627')

# Verify path
python -c "import sys; print(sys.path)"
```

### Permission Errors

**Problem**: Permission denied writing files
```
PermissionError: [Errno 13] Permission denied: 'OptimalWeights/adapter.pth'
```

**Solution**:
```bash
# Fix permissions
chmod -R 755 OptimalWeights/
chmod -R 755 data/

# Change ownership
sudo chown -R $USER:$USER OptimalWeights/

# Use different output directory
OUTPUT_DIR=/tmp/weights python src/Training/train_adapters.py
```

## Debugging Tips

### Enable Verbose Logging
```bash
# Set logging level
export LOG_LEVEL=DEBUG
python src/Training/train_adapters.py

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile Performance
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)
```

### Check Resource Usage
```bash
# Monitor system resources
htop

# Check disk space
df -h

# Monitor GPU
nvidia-smi -l 1

# Check memory usage
free -h
```

## Getting Help

If issues persist:

1. **Check logs**: `Results/adapter_training/latest.log`
2. **Validate environment**: `python scripts/check_environment.py`
3. **Run diagnostics**: `python scripts/diagnose_issues.py`
4. **Review documentation**: [README.md](README.md), [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
5. **Search issues**: GitHub repository issues page
6. **Contact support**: Open new issue with error logs and system info