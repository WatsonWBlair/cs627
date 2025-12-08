# Troubleshooting Guide for pregenerate_tokens.py

## Identified Issues

### 1. **Python Environment Problem**
- **Symptom**: Python scripts hang or crash with segmentation fault when importing PyTorch
- **Cause**: PyTorch 2.9.0 (development version) appears to be incompatible with the current environment
- **Solution**: Downgrade to stable PyTorch version

### 2. **Import Path Issues**
- **Symptom**: Cannot import encoder modules
- **Cause**: Python path configuration or module initialization
- **Solution**: Use the fixed version with better import handling

## Solutions

### Option 1: Use the Fixed Script
A fixed version has been created at `src/Training/pregenerate_tokens_fixed.py` with:
- Better error handling
- Graceful fallbacks for missing encoders
- Detailed diagnostic output

```bash
python src/Training/pregenerate_tokens_fixed.py
```

### Option 2: Fix PyTorch Installation
The current PyTorch 2.9.0 appears to be causing segmentation faults. Reinstall stable version:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install stable version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### Option 3: Use Docker Container
If local environment issues persist, use Docker:

```bash
# Build Docker image with correct dependencies
docker build -t cs627-training .

# Run token generation in container
docker run -v $(pwd)/data:/app/data cs627-training \
  python src/Training/pregenerate_tokens.py
```

### Option 4: Manual Token Generation
If automated script fails, generate tokens manually:

```python
import torch
import numpy as np
import h5py
from src.Encoders.text.semantic_to_vec import Text_to_Vec

# Initialize encoder
encoder = Text_to_Vec()
encoder.eval()

# Process data
texts = ["sample text 1", "sample text 2"]
with torch.no_grad():
    tokens = encoder(texts, pregen=True).cpu().numpy()

# Save to HDF5
with h5py.File('tokens.h5', 'w') as f:
    f.create_dataset('text_base', data=tokens)
```

## Pre-requisites Check

Before running the token generation:

1. **Check MOSI Data**:
```bash
ls -la data/cmumosi/mosi/
ls -la data/cmumosi/audio/
ls -la data/cmumosi/frames/
```

If directories don't exist or are empty:
```bash
# Download MOSI data
python -c "from src.Training.Data_Wrangling.mosi_dataset import download_mosi; download_mosi('data/cmumosi/mosi/')"

# Extract audio/video (if needed)
python scripts/data_wrangling/extract_all_segments.py
```

2. **Check Dependencies**:
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import h5py; print('h5py:', h5py.__version__)"
```

3. **Test Encoder Import**:
```python
# Test if encoders can be imported
from src.Encoders import Text_to_Vec
encoder = Text_to_Vec()
print("Encoder loaded successfully")
```

## Common Error Messages and Solutions

### "Segmentation fault: 11"
- **Cause**: PyTorch version incompatibility
- **Solution**: Reinstall PyTorch or use CPU-only mode:
  ```bash
  DEVICE=cpu python src/Training/pregenerate_tokens.py
  ```

### "No module named 'src'"
- **Cause**: Running script from wrong directory
- **Solution**: Run from project root:
  ```bash
  cd /path/to/cs627
  python src/Training/pregenerate_tokens.py
  ```

### "Dataset is empty"
- **Cause**: MOSI data not downloaded or extracted
- **Solution**: Download and extract data first (see Pre-requisites)

### "CUDA out of memory"
- **Cause**: Batch size too large for GPU
- **Solution**: Reduce batch size:
  ```bash
  BATCH_SIZE=8 python src/Training/pregenerate_tokens.py
  ```

## Alternative: Skip Token Generation

If token generation continues to fail, you can skip it for testing:

1. Create mock tokens for testing adapter training:
```python
import numpy as np
import h5py

# Create fake tokens for testing
num_samples = 100
token_dim = 1024

with h5py.File('data/pregenerated_tokens/mosi/train_tokens.h5', 'w') as f:
    f.create_dataset('text_base', data=np.random.randn(num_samples, token_dim))
    f.create_dataset('audio_waveform', data=np.random.randn(num_samples, token_dim))
    f.create_dataset('segment_ids', data=[f'seg_{i}'.encode() for i in range(num_samples)])
    f.create_dataset('labels', data=np.random.randn(num_samples))
    f.attrs['num_samples'] = num_samples
    f.attrs['token_dim'] = token_dim
    f.attrs['encoders'] = ['text_base', 'audio_waveform']
```

2. Then proceed with adapter training:
```bash
python src/Training/train_adapters.py --mode encoder
```

## Recommended Next Steps

1. **Try the fixed script first**: `pregenerate_tokens_fixed.py`
2. **If that fails**, reinstall PyTorch to stable version
3. **If still failing**, use Docker or manual generation
4. **For quick testing**, use mock tokens

## Contact for Help

If issues persist after trying these solutions:
1. Check GitHub issues: https://github.com/[your-repo]/issues
2. Verify all dependencies match requirements.txt
3. Consider using cloud compute (AWS/GCP) with clean environment