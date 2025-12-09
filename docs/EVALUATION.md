# Evaluation Guide

Guidelines for evaluating encoders, decoders, and the Semantic-Vector Space architecture.

**API reference**: See [src/Evaluation/README.md](../src/Evaluation/README.md) for programmatic usage and metrics.

**Benchmarks**: See [BENCHMARKS.md](BENCHMARKS.md) for benchmark datasets and adapters.

## Evaluation Philosophy

**Core Principle**: Semantic-Vector Space quality is measured by how well semantic meaning is preserved across modality transitions.

**Key Insight**: Traditional modality-specific metrics (BLEU, SSIM, etc.) measure aesthetic quality, but semantic alignment quality requires cross-modal evaluation.

## Encoder Evaluation

### 1. Contrastive Learning Error

Measure triplet loss on held-out validation data using the standard formula:

```
Loss = max(cos_sim(query, positive) - cos_sim(query, negative) + margin, 0)
```

**Success Criteria**: Mean loss < 0.05 on validation set

### 2. Cross-Modal Retrieval

Test if semantically-similar content from different modalities produces similar vectors.

**Process**:
```
Text: "A dog running in the park"  → Text Encoder  → Vector A
Audio: [dog barking sounds]         → Audio Encoder → Vector B
Image: [photo of dog in park]       → Image Encoder → Vector C

Measure: cos_sim(A, B), cos_sim(A, C), cos_sim(B, C)
```

Calculate Recall@K:
- For each text vector, rank all audio vectors by similarity
- Recall@1: % of times the correct audio is ranked first
- Recall@5: % of times the correct audio is in top 5
- Recall@10: % of times the correct audio is in top 10

**Success Criteria**:
- Recall@1 > 0.50
- Recall@5 > 0.80
- Recall@10 > 0.90

### 3. Semantic Clustering

Verify that semantically-related content clusters together in vector space.

**Visualization**:
```
Vector Space (t-SNE projection):

    [Sports vectors clustered here]
            ●●●●●
           ●●●●●●●

                        [Music vectors clustered here]
                                ○○○○○
                               ○○○○○○○

  [Nature vectors clustered here]
         ▲▲▲▲▲
        ▲▲▲▲▲▲▲
```

**Metric**: Silhouette score (higher = better clustering)

**Success Criteria**: Silhouette score > 0.5

## Decoder Evaluation

> **IMPORTANT**: Decoder training uses a **dual-loss approach** that balances reconstruction quality and semantic fidelity. The final decoder loss is:
>
> ```
> Total Loss = α × Reconstruction Loss + β × Semantic Fidelity Loss
> ```
>
> Where:
> - **Reconstruction Loss**: Measures aesthetic/perceptual quality (BLEU, SSIM, etc.)
> - **Semantic Fidelity Loss**: Measures semantic drift (`-cos_sim(original_vec, reconstructed_vec)`)
> - **α, β**: Weights to balance the two objectives
>
> Both losses are critical and must be evaluated separately.

### 1. Reconstruction Quality

Measure aesthetic/perceptual fidelity using modality-specific metrics.

**Reconstruction Pipeline**:
```
Original → Encoder → Semantic Vector → Decoder → Reconstructed
  [data]              [1024-dim]                    [data]
```

**Modality-Specific Metrics**:

**Text**:
- BLEU score (n-gram overlap)
- ROUGE-L (longest common subsequence)
- Success: BLEU > 0.40, ROUGE-L > 0.50

**Images**:
- SSIM (structural similarity)
- PSNR (peak signal-to-noise ratio)
- Success: SSIM > 0.70, PSNR > 25 dB

**Audio**:
- Mel-spectrogram distance
- Success: Spectral distance < 5.0

### 2. Semantic Fidelity (Critical)

Measure semantic drift after encode→decode→encode cycle.

**Round-Trip Test**:
```
Step 1: Original Data → Encoder → Vector A [1024-dim]
Step 2: Vector A → Decoder → Reconstructed Data
Step 3: Reconstructed Data → Encoder → Vector B [1024-dim]
Step 4: Measure cos_sim(Vector A, Vector B)
```

If semantic meaning is preserved, Vector A ≈ Vector B (high cosine similarity).

**Success Criteria**:
- Mean fidelity > 0.85
- Min fidelity > 0.70
- Std fidelity < 0.10

## End-to-End System Evaluation

### 1. Cross-Modal Translation

Test translation chains between modalities.

**Example: Text → Image → Text**
```
Original Text: "A red car on a highway"
      ↓
   [Text Encoder]
      ↓
  Vector A [1024-dim]
      ↓
   [Image Decoder]
      ↓
Generated Image: [🚗 on highway]
      ↓
   [Image Encoder]
      ↓
  Vector B [1024-dim]
      ↓
   [Text Decoder]
      ↓
Reconstructed Text: "A red automobile on the road"

Evaluate: cos_sim(Vector A, Vector B)
```

**Success Criteria**:
- Mean similarity > 0.70
- Translation success rate (similarity > 0.70) > 80%

### 2. Downstream Task Performance

Test semantic vectors on actual NLU tasks by training a simple classifier on frozen encoder features.

**Task Examples**:
- Intent classification
- Sentiment analysis
- Question answering
- Image captioning

**Success Criteria**: Accuracy within 5% of task-specific fine-tuned models

## Evaluation Datasets

### Recommended Datasets

**Text**:
- GLUE benchmark (NLU tasks)
- SQuAD (question answering)
- CLINC150 (intent classification)

**Audio**:
- CMU-MOSI (multimodal sentiment)
- LibriSpeech (speech recognition)
- ESC-50 (environmental sound classification)

**Images**:
- COCO (image captioning)
- ImageNet (classification)
- Conceptual 12M (image-text pairs)

**Cross-Modal**:
- CMU-MOSI (text + audio + video)
- Flickr30k (image + text)
- MELD (multimodal emotion)

## Evaluation Workflow

### Stage 1: Component Evaluation
```
Train Encoder → Evaluate Metrics → Pass? → Proceed to Stage 2
                        ↓ Fail
                  Iterate Training
```

**Metrics to Check**:
1. Contrastive learning error (validation set)
2. Cross-modal retrieval (if paired data available)
3. Semantic clustering

### Stage 2: Decoder Evaluation
```
Train Decoder → Evaluate Metrics → Pass? → Proceed to Stage 3
                        ↓ Fail
                  Iterate Training
```

**Metrics to Check**:
1. Reconstruction quality (modality-specific metrics)
2. Semantic fidelity (encode→decode→encode cycle)

**Iteration Trigger**: Semantic fidelity < 0.85

### Stage 3: System Evaluation
```
Evaluate Cross-Modal Chains → Test Downstream Tasks → Measure Performance
                                                              ↓
                                                    Document & Deploy
```

## Reporting Format

Create evaluation reports with this structure:

```
# Evaluation Report: {Component Name}

## Component Details
- Type: Encoder / Decoder
- Modality: Text / Audio / Image
- Base Model: {model_name}
- Training Date: {date}
- Training Data: {dataset}

## Metrics

### Primary Metrics
- Contrastive Loss (validation): {value}
- Semantic Fidelity: {value}
- Recall@1: {value}

### Secondary Metrics
- BLEU / SSIM / etc: {value}
- Inference Latency: {value} ms
- GPU Memory: {value} MB

## Qualitative Analysis

### Success Cases
- Example 1: ...
- Example 2: ...

### Failure Modes
- Issue 1: ...
- Issue 2: ...

## Recommendations
- Next steps: ...
- Potential improvements: ...
```

## Continuous Evaluation

### After Every Training Run

1. Log all metrics to tracking system
2. Compare against previous best
3. Save model only if improvement detected

### Weekly Evaluation

1. Run full evaluation suite on all components
2. Update evaluation dashboard
3. Identify regression or improvement trends

### Before Deployment

1. Run full evaluation on held-out test set
2. Verify all success criteria met
3. Document any known limitations
4. Get stakeholder approval

## Common Pitfalls

1. **Overfitting to validation set**: Always reserve a final held-out test set
2. **Ignoring semantic fidelity**: High BLEU doesn't mean semantic meaning is preserved
3. **Cherry-picking examples**: Report aggregate statistics, not just best cases
4. **Comparing across datasets**: Only compare models evaluated on same data
5. **Ignoring inference cost**: A 1% accuracy gain isn't worth 10x latency increase

## Tools and Libraries

**Metrics**:
- `evaluate` (HuggingFace) - NLP metrics
- `scikit-image` - Image quality metrics
- `librosa` - Audio analysis
- `sentence-transformers` - Embedding evaluation

**Tracking**:
- `wandb` - Experiment tracking
- `tensorboard` - Visualization
- `mlflow` - Model registry

**Datasets**:
- `datasets` (HuggingFace) - Standard benchmarks
- `mteb` - Embedding benchmarks
- `CMU-MultimodalSDK` - Multimodal datasets
