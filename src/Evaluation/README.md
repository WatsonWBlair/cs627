# Evaluation Infrastructure

Comprehensive evaluation tools for cross-modal encoder alignment to the shared semantic vector space.

**Philosophy & methodology**: See [docs/EVALUATION.md](../../docs/EVALUATION.md) for evaluation concepts and success criteria.

**Benchmarks**: See [benchmarks/README.md](benchmarks/README.md) for MTEB, GLUE, and MultiBench adapters.

## Overview

This module provides:
1. **Metrics computation** - Recall@K, cosine similarity, alignment quality, modality gap (MMD)
2. **Cross-modal evaluator** - High-level API for encoding datasets and computing metrics
3. **MultiBench adapter** - Integration with MultiBench benchmark suite for downstream tasks

## Quick Start

### Basic Cross-Modal Evaluation

```bash
# Evaluate on test split with all encoders
python scripts/run_evaluation.py

# Evaluate on specific split with sample limit
python scripts/run_evaluation.py --split train --max-samples 100

# Save embeddings for later analysis
python scripts/run_evaluation.py --save-embeddings
```

### Programmatic Usage

```python
from src.Evaluation import CrossModalEvaluator
from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset

# Load trained encoders
encoders = {
    "text": Text_to_Vec(),
    "audio": Audio_to_Vec(),
    "image": Image_to_Vec()
}

# Load dataset
dataset = MOSIRawVideoDataset(split='test')

# Run evaluation
evaluator = CrossModalEvaluator(encoders)
results = evaluator.evaluate(
    dataset,
    batch_size=32,
    k_values=[1, 5, 10],
    save_results='results/eval_metrics.json'
)
```

## Components

### 1. Metrics (`metrics.py`)

Core metric computation functions:

- **`compute_recall_at_k()`** - Cross-modal retrieval accuracy
  - For each query, rank targets by similarity
  - Check if ground-truth target appears in top-K results
  - Returns R@1, R@5, R@10

- **`compute_cosine_similarity_matrix()`** - Pairwise similarity between modalities
  - Computes mean cosine similarity for all modality pairs
  - Useful for visualizing alignment quality

- **`compute_alignment_quality()`** - Consistency of cross-modal alignment
  - Measures variance of within-pair distances
  - Lower variance = more consistent alignment

- **`compute_modality_gap()`** - Distribution shift between modalities
  - Uses Maximum Mean Discrepancy (MMD) with RBF kernel
  - Lower MMD = better alignment to shared space

- **`compute_cross_modal_metrics()`** - Main entry point
  - Computes all metrics in one call
  - Returns comprehensive results dict

### 2. Cross-Modal Evaluator (`evaluators/cross_modal_evaluator.py`)

High-level evaluator for datasets:

```python
evaluator = CrossModalEvaluator(encoders, device='cuda')

# Encode entire dataset
embeddings = evaluator.encode_dataset(dataset, batch_size=32)

# Run full evaluation
results = evaluator.evaluate(
    dataset,
    max_samples=500,
    save_results='results/metrics.json'
)

# Evaluate specific pairs only
pairs = [("text", "audio"), ("text", "image")]
results = evaluator.evaluate_pairs(dataset, pairs)
```

### 3. MultiBench Adapter (`multibench_adapter.py`)

Adapter for downstream task evaluation:

- Freezes pretrained encoders
- Trains small task-specific heads
- Measures semantic vector space quality

```python
from src.Evaluation.multibench_adapter import MOSISentimentTask

# Create sentiment task adapter
adapter = MOSISentimentTask(
    encoders=encoders,
    fusion_method="concat",  # or "mean", "attention"
    hidden_dim=128,
    freeze_encoders=True
)

# Train on downstream task
history = adapter.train_on_task(
    train_loader,
    val_loader,
    num_epochs=20,
    save_path='results/task_head.pth'
)
```

## Evaluation Metrics

### Cross-Modal Retrieval (Recall@K)

Measures how well encoders align modalities. For each query, rank all targets by similarity:

- **R@1**: Percent of queries where ground-truth is rank 1
- **R@5**: Percent of queries where ground-truth is in top-5
- **R@10**: Percent of queries where ground-truth is in top-10

**Expected values:**
- Random baseline: R@1 = 1/N (e.g., 0.01 for N=100)
- Perfect alignment: R@1 = 1.0
- Good alignment: R@1 > 0.5, R@10 > 0.8

### Cosine Similarity Matrix

Heatmap showing pairwise similarities:

```
           text   audio  image
text       1.00   0.65   0.58
audio      0.65   1.00   0.51
image      0.58   0.51   1.00
```

- Diagonal = 1.0 (self-similarity)
- Off-diagonal: Higher = better alignment
- **Expected:** 0.5-0.7 for aligned modalities

### Alignment Quality

Measures consistency of cross-modal alignment:

```python
{
  "audio": {
    "mean_distance": 0.42,    # Lower is better
    "variance": 0.015,        # Lower = more consistent
    "alignment_score": 0.985  # Higher is better
  }
}
```

- **mean_distance**: Average cosine distance from reference
- **variance**: Consistency of alignment (lower = better)
- **alignment_score**: 1 / (1 + variance)

### Modality Gap (MMD)

Maximum Mean Discrepancy between modality distributions:

```python
{
  "text_audio": 0.0023,   # Lower is better
  "text_image": 0.0031,
  "audio_image": 0.0019
}
```

- Lower MMD = distributions are closer
- **Expected:** < 0.01 for well-aligned encoders

## Command Line Options

```bash
python scripts/run_evaluation.py [OPTIONS]
```

### Dataset Options
- `--split {train,valid,test}` - Which split to evaluate on (default: test)
- `--mosi-data-path PATH` - Path to MOSI metadata (default: data/cmumosi/mosi/)
- `--audio-dir PATH` - Path to audio files (default: data/cmumosi/audio/)
- `--video-dir PATH` - Path to video frames (default: data/cmumosi/frames/)

### Evaluation Options
- `--batch-size N` - Batch size for encoding (default: 32)
- `--max-samples N` - Limit evaluation to N samples (default: all)
- `--k-values K [K ...]` - K values for Recall@K (default: 1 5 10)

### Output Options
- `--output-dir DIR` - Where to save results (default: results/evaluation/)
- `--save-embeddings` - Save encoded embeddings to disk

### Encoder Selection
- `--encoders E [E ...]` - Which encoders to use (default: text audio_waveform audio_tone image)

### Downstream Tasks
- `--downstream-task {mosi_sentiment}` - Run downstream task evaluation (optional)
- `--task-epochs N` - Epochs for task training (default: 20)

## Output Files

Evaluation creates timestamped directory with:

```
results/evaluation/eval_test_20250103_123456/
├── config.json                    # Evaluation configuration
├── cross_modal_metrics.json       # All evaluation metrics
└── embeddings.pt                  # Encoded embeddings (if --save-embeddings)
```

### Metrics JSON Structure

```json
{
  "retrieval": {
    "text_audio": {"R@1": 0.65, "R@5": 0.85, "R@10": 0.92},
    "text_image": {"R@1": 0.58, "R@5": 0.78, "R@10": 0.88},
    ...
  },
  "similarity_matrix": {
    "matrix": [[1.0, 0.65, 0.58], ...],
    "modality_names": ["text", "audio", "image"]
  },
  "alignment_quality": {
    "audio": {"mean_distance": 0.42, "variance": 0.015, ...},
    "image": {"mean_distance": 0.51, "variance": 0.023, ...}
  },
  "modality_gap": {
    "text_audio": 0.0023,
    "text_image": 0.0031,
    ...
  },
  "intra_modal_variance": {
    "text": {"mean_dist_from_centroid": 0.31, ...},
    ...
  }
}
```

## Integration with Training

After training encoders:

```bash
# 1. Train encoders
python src/Training/train_encoders.py

# 2. Evaluate alignment
python scripts/run_evaluation.py --split test

# 3. Check results
cat results/evaluation/eval_test_*/cross_modal_metrics.json
```

## Next Steps

### Phase 2: Adapter Ablation Study

See `ABLATION_STUDY.md` (coming soon) for:
- Bayesian optimization of adapter hyperparameters
- Hidden layer experiments (1-5 layers)
- Hidden dimension experiments (64-512 units)
- 15 configurations evaluated with Gaussian Process

### Phase 3: Matrix Factorization

See `FACTORIZATION_ANALYSIS.md` (coming soon) for:
- SVD analysis of semantic vector space
- NMF decomposition for interpretable factors
- Principal component analysis
- Visualization of learned structure

## Troubleshooting

### "No embeddings were generated"
- Check that dataset has samples: `len(dataset) > 0`
- Verify encoders loaded correctly: Check for adapter weight warnings
- Ensure data directories exist and contain files

### "Unsupported data type"
- Dataset must return dict with keys: 'text', 'audio', 'video'
- Text should be list of strings
- Audio/video should be tensors

### Low Recall@K scores
- Check training converged: Review training_reports/loss_curves.png
- Verify data quality: Inspect sample data with dataset[0]
- Consider more training epochs or larger batch size

## References

- Liang et al. "MultiBench: Multiscale Benchmarks for Multimodal Representation Learning" (NeurIPS 2021)
- Gretton et al. "A Kernel Two-Sample Test" (JMLR 2012)
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
