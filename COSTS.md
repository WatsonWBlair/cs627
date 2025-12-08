# CS627 Pipeline Cost Breakdown

## Summary
**Complete training pipeline: $5-10** | **6-month research: ~$120-150** | **Production: ~$760/month**

## Training Costs

### Phase 1: Token Pre-generation
- **Purpose**: Generate encoder outputs once (10-100x speedup)
- **Time**: ~1 hour on g5.4xlarge
- **Cost**: $1.62
- **Storage**: ~200MB HDF5 files

### Phase 2: Adapter Training  
- **Purpose**: Train lightweight MLPs (7.7M params vs 619M full)
- **Time**: ~1 hour total
- **Cost**: $1.62
- **Memory**: <2GB VRAM (adapters only)

## AWS Infrastructure

### Recommended: g5.4xlarge
| Component | Specification | Cost |
|-----------|--------------|------|
| Instance | 16 vCPUs, 64GB RAM | $1.62/hour |
| GPU | A10G (24GB VRAM) | Included |
| Storage | 200GB gp3 SSD | $16/month |

### Alternative Options
| Instance | GPU | VRAM | Cost/Hour | Use Case |
|----------|-----|------|-----------|----------|
| g5.xlarge | A10G | 24GB | $1.01 | Budget |
| g4dn.xlarge | T4 | 16GB | $0.53 | Testing |
| Spot Instances | - | - | -70% | Cost savings |

## Storage & Data

### Training Data
- **MOSI Dataset**: 2GB (~$0.05/month S3)
- **Pre-generated Tokens**: 200MB
- **Model Weights**: 100MB (adapters only)
- **S3 Transfer**: ~$0.09/GB

## Inference Costs

### Production Deployment
- **Latency**: <200ms per request
- **Memory**: 8GB VRAM (all encoders)
- **Server**: g5.xlarge @ $730/month
- **API Cost**: ~$0.01-0.05 per 1000 requests

## Optimization Strategies

1. **Pre-generated Tokens**: 10-100x training speedup
2. **Spot Instances**: 70% cost reduction
3. **Mixed Precision**: 50% memory savings
4. **Early Stopping**: Reduce unnecessary epochs
5. **Adapter-only Training**: 100x parameter reduction

## Model Sizes

| Model | Parameters | VRAM |
|-------|------------|------|
| BART Text Encoder | 140M | 2GB |
| Whisper Audio | 244M | 3GB |
| WavLM Tone | 95M | 1.5GB |
| ViT-GPT2 Image | 140M | 2GB |
| **Total Encoders** | **619M** | **8.5GB** |
| **Trainable Adapters** | **7.7M** | **<100MB** |

## Cost Comparison

| Approach | Parameters | Training Time | Cost |
|----------|------------|---------------|------|
| Full Fine-tuning | 619M | 100+ hours | $500+ |
| Adapter Training (Ours) | 7.7M | 2 hours | $5-10 |
| **Savings** | **99%** | **98%** | **98%** |