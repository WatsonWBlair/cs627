# AWS Deployment Guide for CS627 Training

## Current Test Results (December 4, 2024)

### ✅ GPU Instance Configuration (44.202.102.201)
- **AMI**: Amazon Linux with NVIDIA drivers
- **Instance Type**: g5.4xlarge  
- **GPU**: NVIDIA A10G (23.7 GB VRAM) ✅ Detected
- **PyTorch**: 2.5.1+cu121 ✅ CUDA enabled
- **Training Script**: ✅ Ready to run (detects CUDA automatically)
- **Status**: **READY FOR TRAINING**

### Previous GPU Instance (13.221.100.101)
- **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)
- **Instance Type**: g5.4xlarge
- **GPU**: NVIDIA A10G (23.7 GB VRAM) ✅ Detected
- **Storage**: 130 GB
- **PyTorch**: 2.5.1+cu121 ✅ CUDA enabled

### Test Results on GPU Instance
- **Text Encoder Tests**: 1/3 PASSED
  - ❌ Empty string test: CUDA out of memory
  - ✅ Initialization test: PASSED
  - ❌ String input test: CUDA out of memory
- **Audio Encoder Tests**: 1/2 PASSED
  - ✅ Initialization test: PASSED  
  - ❌ Waveform input test: CUDA out of memory
- **Image Encoder Tests**: 0/2 PASSED
  - ❌ Initialization test: PyTorch version issue (needs 2.6+)
  - ❌ PIL image test: PyTorch version issue
- **Consistency Test**: ❌ FAILED

### Issues Found
1. **CUDA Memory**: Multiple models exceed GPU memory when loaded simultaneously
2. **PyTorch Version**: Image encoder requires PyTorch 2.6+ (have 2.5.1)
3. **Solution**: Need to either:
   - Upgrade PyTorch to 2.6+
   - Load models sequentially instead of simultaneously
   - Use model offloading strategies

## Current Test Results (December 4, 2024) - Earlier Attempt

### ❌ Failed Instance Configuration
- **AMI**: ami-042b823591be39990 (Amazon Linux 2023)
- **Instance IP**: 35.175.222.131
- **Issues**:
  - Disk completely full (30G, 100% used)
  - No GPU detected
  - Dependency installation failures
  - Model download failures (insufficient space)

### Test Results
- **Text Encoder Tests**: ✅ PASSED (3/3)
- **Audio Encoder Tests**: ❌ FAILED (0/2) - Insufficient disk space for model download
- **Image Encoder Tests**: ❌ FAILED (0/2) - Insufficient disk space for model download
- **Consistency Tests**: ❌ FAILED - Could not load all encoders

## ✅ Recommended Configuration

Launch the correct AMI with proper specifications:

```bash
aws ec2 run-instances \
  --image-id ami-04f3e35dc85e9423b \
  --instance-type g5.4xlarge \
  --key-name SHARD_Training \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=CS627-Training}]'
```

### Key Requirements:
1. **AMI**: `ami-04f3e35dc85e9423b` (AWS Deep Learning OSS Nvidia Driver AMI)
   - Pre-installed: PyTorch 2.5, CUDA 12.4, all ML dependencies
   - No driver installation needed
   - No Python environment setup required
   
2. **Instance Type**: `g5.4xlarge`
   - NVIDIA A10G GPU (24GB VRAM)
   - Required for training
   
3. **Storage**: 200GB gp3 SSD
   - Current instance only has 30GB (insufficient)
   - Models require ~3GB just for download
   - Training data requires ~60GB

## Deployment Script

Once the correct instance is launched:

```bash
# Deploy code and data
./scripts/setup_remote_instance.sh <new-instance-ip>

# SSH and start training
ssh -i ~/.ssh/SHARD_Training.pem ubuntu@<new-instance-ip>
cd ~/cs627
python src/Training/train_encoders.py
```

## Training Entry Point

**Main Script**: `src/Training/train_encoders.py`

Features:
- Automatic GPU detection
- Multi-encoder training (Text, Audio, Image, Tone)
- Momentum contrastive learning
- Training reporter with visualizations
- Automatic weight saving to `OptimalWeights/`
- Environment variable configuration support

## Troubleshooting

### Issue: Disk Space
- **Symptom**: "Not enough free disk space" warnings
- **Solution**: Use 200GB+ storage volume

### Issue: Wrong AMI
- **Symptom**: Manual Python/CUDA installation required
- **Solution**: Use Deep Learning AMI (ami-04f3e35dc85e9423b)

### Issue: No GPU
- **Symptom**: Training runs on CPU (very slow)
- **Solution**: Use g5.4xlarge or similar GPU instance

## Next Steps

1. Terminate current instance (35.175.222.131)
2. Launch new instance with recommended configuration
3. Deploy code using setup script
4. Run full test suite
5. Start training if tests pass