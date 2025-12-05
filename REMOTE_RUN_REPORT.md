# Remote Deployment and Training Report

## Overview
This report documents the deployment of CS627 Semantic-Vector Space training to AWS EC2 instance (54.146.55.195) with the challenges encountered and solutions implemented.

## Deployment Summary
- **Instance**: AWS EC2 with NVIDIA A10G GPU (ami-019056869a13971ff)
- **Docker Image Size**: 8.43GB
- **Data Transferred**: 2.9GB CMU-MOSI dataset
- **Training Target**: 70 epochs (reduced from 100) for both encoders and decoders
- **Status**: Encoder training in progress after multiple remediation steps

## Major Issues and Resolutions

### 1. CMU-MultimodalSDK Download Blocking (Critical)

**Error**: 
```
RuntimeError: CMU_MOSI_TimestampedWords.csd file already exists...
```

**Root Cause**: CMU-MultimodalSDK's `download()` function unconditionally tries to download files even if they already exist. When files are present, it raises an exception instead of using them.

**Solution**: Created `train_encoders_patched.py` that bypasses the SDK entirely:
- Implemented `SimpleMOSIDataset` class that directly reads audio/video files from disk
- Removed dependency on SDK's complex data loading mechanism
- Maintained compatibility with existing training pipeline

### 2. Docker Build Location Issue

**Error**: Attempted to build Docker image on remote server without source files

**Root Cause**: Misunderstanding of deployment workflow

**Solution**: 
- Built Docker image locally (8.43GB)
- Saved as tar archive: `docker save cs627-training:latest > cs627-training-with-weights.tar`
- Transferred to remote via SCP (~28 minutes)

### 3. Missing Pre-trained Weights in Docker

**Error**: OptimalWeights directory not included in Docker image

**Root Cause**: Dockerfile didn't include COPY command for weights

**Solution**: Added to Dockerfile:
```dockerfile
COPY OptimalWeights/ /workspace/OptimalWeights/
```

### 4. Data Transfer Strategy

**Challenge**: 2.9GB MOSI dataset needed on remote

**Initial Consideration**: SSH-based volume mounting (not feasible)

**Solution**: One-time data transfer
- Compressed locally: `tar -czf cmumosi_data.tar.gz data/cmumosi/`
- Transferred via SCP (~35 minutes)
- Extracted on remote: `tar -xzf cmumosi_data.tar.gz`

### 5. Torch Version Compatibility

**Error**: 
```
ImportError: cannot import name 'MultiScaleDeformableAttention' from 'groundingdino.models.GroundingDINO.csrc' (requires torch v2.6)
```

**Root Cause**: Image encoder requires torch 2.6, but Docker has torch 2.5

**Solution**: Removed image encoder from training, focusing on text-audio alignment

### 6. Audio Encoder Input Format Issues

**Error 1**: `forward_audio()` method doesn't exist
```python
audio_encoder.forward_audio(batch['audio'])  # Wrong
```

**Solution**: Use standard forward method:
```python
audio_encoder(batch['audio'])  # Correct
```

**Error 2**: CUDA tensor conversion error
```
TypeError: can't convert cuda:0 device type tensor to numpy
```

**Solution**: Audio encoder expects CPU tensors/numpy arrays, not CUDA tensors

**Error 3**: Shape broadcasting error in Whisper processor
```
ValueError: operands could not be broadcast together with remapped shapes
```

**Solution**: Modified collate function to return list of numpy arrays instead of stacked tensors:
```python
'audio': audio_arrays,  # List of numpy arrays for Whisper
# Instead of:
'audio': torch.stack(audio_tensors)
```

### 7. SSH Authentication

**Error**: Wrong username for AMI
```
ubuntu@54.146.55.195: Permission denied
```

**Solution**: Use correct username for Amazon Linux:
```bash
ssh -i ~/.ssh/aws_scalable.pem ec2-user@54.146.55.195
```

## Timing Summary

| Task | Duration | Status |
|------|----------|--------|
| Docker Build (local) | ~45 minutes | Completed |
| Docker Transfer | 28 minutes | Completed |
| Data Transfer | 35 minutes | Completed |
| Factorization Study | Pending | Blocked by SDK |
| Encoder Training (70 epochs) | In Progress | Running with patched script |
| Decoder Training (70 epochs) | Pending | Waiting for encoders |

## Key Learnings

1. **SDK Dependencies**: Production training scripts should avoid complex SDK dependencies that can fail in containerized environments
2. **Data Strategy**: For large datasets, one-time transfer is more practical than complex mounting solutions
3. **Version Compatibility**: Docker images must carefully manage torch versions when using cutting-edge models
4. **Input Formats**: Different models expect different input formats (tensors vs numpy arrays)
5. **AMI Selection**: Deep Learning AMIs (ami-019056869a13971ff) provide pre-configured GPU environments

## Recommendations

1. **Update Main Training Script**: Incorporate the simplified dataset loader into the main training pipeline
2. **Version Pinning**: Update Docker to torch 2.6+ for full model compatibility
3. **SDK Alternative**: Consider replacing CMU-MultimodalSDK with direct file access
4. **Error Recovery**: Add better error handling and automatic retry mechanisms
5. **Progress Monitoring**: Implement better logging for long-running training jobs

## Current Status

As of the last update:
- Docker container successfully deployed to remote
- Data successfully transferred and mounted
- Encoder training running with patched script (70 epochs)
- Text and Audio encoders training (Image encoder skipped due to version issues)
- Monitoring training progress via SSH

## Next Steps

1. Complete encoder training (70 epochs)
2. Run decoder training (70 epochs)
3. Execute factorization study with patched script
4. Collect and analyze training metrics
5. Transfer trained weights back to local environment

---

*Report generated: December 5, 2024*
*Project: CS627 Semantic-Vector Space*
*Deployment Target: AWS EC2 (54.146.55.195)*