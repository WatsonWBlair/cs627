# Benchmarks Overview

Benchmark datasets and evaluation metrics for the Semantic-Vector Space project.

**Technical details**: See [src/Evaluation/benchmarks/README.md](../src/Evaluation/benchmarks/README.md) for adapter APIs and code examples.

**Related**: See [EVALUATION.md](EVALUATION.md) for evaluation philosophy and methodology.

## Benchmark Suite

| Benchmark | Tasks | Focus | Adapter |
|-----------|-------|-------|---------|
| **MTEB** | 56+ | Text embeddings | `mteb_adapter.py` |
| **GLUE** | 9 | Language understanding | `glue_adapter.py` |
| **MultiBench** | 15 datasets | Multimodal learning | `multibench_adapter.py` |
| **CMU-MOSI** | 1 | Cross-modal alignment | Built-in |

## Quick Evaluation

```bash
# Run comprehensive evaluation
python scripts/evaluation/run_full_evaluation.py

# Quick MTEB subset
python -c "
from src.Encoders import Text_to_Vec
from src.Evaluation.benchmarks.mteb_adapter import run_mteb_evaluation
results = run_mteb_evaluation(Text_to_Vec(), quick_eval=True)
print(f'Average: {results[\"average\"]:.3f}')
"
```

## Key Metrics

### Cross-Modal Metrics

| Metric | Good | Excellent | Description |
|--------|------|-----------|-------------|
| **R@1** | >0.5 | >0.7 | Top-1 retrieval accuracy |
| **R@5** | >0.8 | >0.9 | Top-5 retrieval accuracy |
| **MRR** | >0.7 | >0.8 | Mean reciprocal rank |
| **Modality Gap** | <0.2 | <0.1 | Distribution distance (MMD) |

### Text Embedding Metrics

| Metric | Good | Excellent | Tasks |
|--------|------|-----------|-------|
| **Spearman** | >0.8 | >0.9 | STS similarity |
| **V-Measure** | >0.6 | >0.7 | Clustering |
| **MAP** | >0.5 | >0.7 | Retrieval |

### Decoder Metrics

| Metric | Good | Excellent | Description |
|--------|------|-----------|-------------|
| **BLEU-4** | >0.3 | >0.5 | N-gram precision |
| **Semantic Fidelity** | >0.85 | >0.95 | Encode→decode preservation |

## Performance Tiers

| Tier | Cross-Modal R@1 | MTEB Avg | GLUE Avg |
|------|-----------------|----------|----------|
| **SOTA** | >0.70 | >0.75 | >0.85 |
| **Strong** | >0.60 | >0.65 | >0.75 |
| **Good** | >0.50 | >0.55 | >0.65 |
| **Baseline** | >0.30 | >0.45 | >0.55 |

## Expected Results by Training Stage

| Stage | Epochs | R@1 | MTEB | GLUE |
|-------|--------|-----|------|------|
| Initial | 1-5 | 0.3-0.4 | 0.45-0.55 | 0.60-0.70 |
| Mid | 5-20 | 0.5-0.6 | 0.55-0.65 | 0.70-0.75 |
| Converged | 20+ | 0.65-0.75 | 0.65-0.75 | 0.75-0.85 |

## Baseline Comparisons

| Model | MTEB Avg | GLUE Avg | Our Advantage |
|-------|----------|----------|---------------|
| all-MiniLM-L6-v2 | 63.0 | 77.0 | Multimodal support |
| all-mpnet-base-v2 | 69.7 | 82.0 | Lighter adapters |
| e5-large-v2 | 75.4 | 85.0 | Modular design |

## Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low R@1 (<0.3) | Poor alignment | Increase batch size, adjust temperature |
| MTEB <0.5 | Limited text data | Add MTEB/GLUE training data |
| High modality gap | Unbalanced modalities | Balance sampling, add alignment loss |
| Low decoder fidelity | Semantic drift | Increase semantic loss weight |

## Documentation

- **Adapter APIs**: [src/Evaluation/benchmarks/README.md](../src/Evaluation/benchmarks/README.md)
- **Evaluation guide**: [EVALUATION.md](EVALUATION.md)
- **Training workflow**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## References

- [MTEB Paper](https://arxiv.org/abs/2210.07316) | [Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [GLUE Paper](https://arxiv.org/abs/1804.07461) | [Leaderboard](https://gluebenchmark.com/leaderboard)
- [MultiBench Paper](https://arxiv.org/abs/2107.07502)
- [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
