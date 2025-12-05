# Cloud Training Guide - AWS Setup Without mmsdk

## Problem Solved
The original `train_encoders.py` script requires the `mmsdk` package (CMU-MultimodalSDK) which is not available via pip. This guide provides a standalone training solution that works directly with the transferred data files.

## Solution: Standalone Training Script

### 1. Use `train_encoders_standalone.py`
Located at: `src/Training/train_encoders_standalone.py`

**Key Features:**
- Works without mmsdk dependency
- Loads audio/video files directly from disk
- Implements simple contrastive learning
- Handles CUDA memory efficiently
- Saves adapter weights automatically

### 2. AWS Instance Setup

#### Launch Instance
```bash
# Recommended: g5.4xlarge with NVIDIA A10G GPU
# Any Amazon Linux or Ubuntu AMI with GPU support
```

#### Connect to Instance
```bash
ssh -i ~/.ssh/SHARD_Training.pem ec2-user@<instance-ip>
# or
ssh -i ~/.ssh/SHARD_Training.pem ubuntu@<instance-ip>
```

### 3. Quick Setup Commands

```bash
# On local machine - deploy code and data
./scripts/setup_remote_instance.sh <instance-ip>

# On remote instance - setup Python environment
cd ~/cs627
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML dependencies from cloud_requirements.txt
pip install transformers huggingface-hub safetensors librosa matplotlib numpy scipy scikit-learn pytest tqdm h5py python-dotenv
```

### 4. Run Training

#### Basic Training
```bash
cd ~/cs627
source venv/bin/activate
export PYTHONPATH=/home/ec2-user/cs627:/home/ec2-user/cs627/src:$PYTHONPATH
python src/Training/train_encoders_standalone.py
```

#### Custom Parameters
```bash
# Adjust training parameters via environment variables
EPOCHS=10 BATCH_SIZE=16 LEARNING_RATE=0.0001 python src/Training/train_encoders_standalone.py
```

### 5. Verify Training

```bash
# Check saved weights
ls -la OptimalWeights/

# Expected output:
# facebook_bart-base_text_enc_weights.pth
# openai_whisper-small_audio_enc_weights.pth
```

### 6. Test Setup

```bash
# Run quick test to verify everything works
python scripts/quick_train_test.py

# Run comprehensive test
python scripts/test_cloud_setup.py
```

## Training Results

### Successfully Tested Configuration
- **Instance**: g5.4xlarge
- **GPU**: NVIDIA A10G (23.7 GB VRAM)
- **PyTorch**: 2.5.1+cu121
- **Training Time**: ~7 seconds per epoch (100 samples, batch size 4)
- **GPU Memory Usage**: ~1.6 GB allocated

### Loss Progression Example
- Epoch 1: 1.3829
- Epoch 2: 1.3442

## Troubleshooting

### Issue: ModuleNotFoundError for Encoders
**Solution**: Set PYTHONPATH correctly
```bash
export PYTHONPATH=/home/ec2-user/cs627:/home/ec2-user/cs627/src:$PYTHONPATH
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size
```bash
BATCH_SIZE=2 python src/Training/train_encoders_standalone.py
```

### Issue: Image encoder fails (PyTorch < 2.6)
**Solution**: The script will continue with text and audio encoders only. To fix:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

## Files Created for Cloud Training

1. **`cloud_requirements.txt`** - Minimal package list for cloud
2. **`train_encoders_standalone.py`** - Training script without mmsdk
3. **`test_cloud_setup.py`** - Comprehensive setup verification
4. **`quick_train_test.py`** - Quick encoder test

## Data Requirements

The standalone script expects:
- Audio files in: `data/cmumosi/audio/*.wav`
- Video frames in: `data/cmumosi/frames/*.jpg`

Files are matched by filename stem (e.g., `video_1.wav` matches with `video_1.jpg`).

## Next Steps

1. **Scale up training**: Increase epochs and batch size
2. **Add validation**: Split data into train/val sets
3. **Monitor metrics**: Add tensorboard or wandb logging
4. **Optimize hyperparameters**: Adjust learning rate, temperature, etc.
5. **Add more modalities**: Enable image encoder with PyTorch 2.6+