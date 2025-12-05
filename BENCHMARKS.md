# Benchmarks Documentation

Detailed information about benchmark datasets, evaluation metrics, and interpretation guidelines.

**Related**: See [EVALUATION.md](EVALUATION.md) for evaluation philosophy and methodology.

## Table of Contents

1. [Overview](#overview)
2. [Benchmark Datasets](#benchmark-datasets)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Running Evaluations](#running-evaluations)
5. [Interpreting Results](#interpreting-results)
6. [Performance Baselines](#performance-baselines)
7. [Troubleshooting](#troubleshooting)

## Overview

Our evaluation framework uses multiple complementary benchmarks to assess encoder and decoder quality:

- **Cross-Modal Evaluation**: Measures alignment between text, audio, and image modalities
- **MTEB (Massive Text Embedding Benchmark)**: 56+ tasks for text embedding evaluation
- **GLUE (General Language Understanding Evaluation)**: 9 NLU tasks for language understanding
- **MultiBench**: Multimodal learning benchmark suite
- **Custom Metrics**: Semantic fidelity, modality gap, and reconstruction quality

## Benchmark Datasets

### 1. CMU-MOSI (Multimodal Opinion Sentiment Intensity)

**Description**: Real-world multimodal sentiment analysis dataset
- **Size**: 2,199 opinion video segments
- **Modalities**: Text transcripts, audio (16kHz), video frames
- **Labels**: Sentiment intensity [-3, 3]
- **Splits**: Train (1,284), Valid (229), Test (686)

**Usage**:
```python
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
dataset = MOSIRawVideoDataset(split='test')
```

**Evaluation Focus**: Cross-modal alignment, sentiment prediction

### 2. MTEB (Massive Text Embedding Benchmark)

**Description**: Comprehensive text embedding evaluation across 8 task categories
- **Tasks**: 56+ diverse tasks
- **Languages**: 112 languages
- **Categories**:
  - **Bitext Mining**: Parallel sentence retrieval
  - **Classification**: Text categorization (Banking77, Amazon Reviews, etc.)
  - **Clustering**: Document grouping (ArXiv, Reddit, StackExchange)
  - **Pair Classification**: Sentence pair relations (QQP duplicates, TwitterSemEval)
  - **Reranking**: Query-document reranking
  - **Retrieval**: Information retrieval (MSMARCO, NaturalQuestions, HotpotQA)
  - **STS**: Semantic Textual Similarity (STS12-22, STSBenchmark)
  - **Summarization**: Text summarization evaluation

**Usage**:
```python
from src.Evaluation.benchmarks.mteb_adapter import MTEBEvaluator
evaluator = MTEBEvaluator(encoder)
results = evaluator.run(tasks=['STS12', 'Banking77Classification'])
```

**Key Tasks for Our Architecture**:
- **STS Tasks**: Direct measure of semantic similarity
- **Retrieval Tasks**: Cross-modal retrieval analogue
- **Clustering Tasks**: Semantic grouping quality

### 3. GLUE (General Language Understanding Evaluation)

**Description**: Standard benchmark for NLU systems
- **Tasks**: 9 diverse language understanding tasks

| Task | Type | Size | Metric | Description |
|------|------|------|--------|-------------|
| CoLA | Single | 10.7k | Matthews | Grammatical acceptability |
| SST-2 | Single | 70k | Accuracy | Sentiment analysis |
| MRPC | Pair | 5.8k | F1/Acc | Paraphrase detection |
| QQP | Pair | 404k | F1/Acc | Question duplicate detection |
| STS-B | Pair | 8.6k | Pearson/Spearman | Semantic similarity |
| MNLI | Pair | 433k | Accuracy | Natural language inference |
| QNLI | Pair | 115k | Accuracy | Question-answering NLI |
| RTE | Pair | 3k | Accuracy | Textual entailment |
| WNLI | Pair | 852 | Accuracy | Coreference resolution |

**Usage**:
```python
from src.Evaluation.benchmarks.glue_adapter import GLUEEvaluator
evaluator = GLUEEvaluator(encoder)
results = evaluator.run(tasks=['sst2', 'mrpc', 'stsb'])
```

**Training Data Extraction**:
```bash
# Extract training triplets from GLUE
python scripts/data_wrangling/wrangle_glue_data.py
```

### 4. MultiBench (Multimodal Learning Benchmark)

**Description**: Comprehensive multimodal learning evaluation
- **Datasets**: 15 datasets across 10 modalities
- **Tasks**: Classification, regression, retrieval
- **Unique Features**: Robustness testing, computational efficiency metrics

**Key Datasets**:
- **AV-MNIST**: Audio-visual digit recognition
- **MOSI/MOSEI**: Multimodal sentiment analysis
- **UR-FUNNY**: Multimodal humor detection
- **MuST-C**: Multilingual speech translation
- **Kinetics-Sounds**: Audio-visual action recognition

**Usage**:
```python
from src.Evaluation.benchmarks.multibench_adapter import MultiBenchEvaluator
encoders = {'text': Text_to_Vec(), 'audio': Audio_to_Vec(), 'image': Image_to_Vec()}
evaluator = MultiBenchEvaluator(encoders)
results = evaluator.run(tasks=['mosi_sentiment'])
```

**Evaluation Focus**: Multimodal fusion, cross-modal understanding, robustness

### 5. Conceptual Captions / Conceptual 12M

**Description**: Large-scale image-text pairs
- **Size**: ~12 million image-alt-text pairs
- **Use Case**: Cross-modal retrieval, zero-shot transfer
- **Quality**: Automatically collected and filtered

## Evaluation Metrics

### Cross-Modal Metrics

#### 1. Recall@K (R@K)
**Definition**: Percentage of queries where the correct item appears in top K results

**Formula**:
```
R@K = (# queries with correct match in top K) / (total # queries)
```

**Interpretation**:
- R@1 > 0.5: Good alignment
- R@5 > 0.8: Strong alignment
- R@10 > 0.9: Excellent alignment

#### 2. Mean Reciprocal Rank (MRR)
**Definition**: Average of reciprocal ranks of correct items

**Formula**:
```
MRR = (1/|Q|) × Σ(1/rank_i)
```

**Interpretation**:
- MRR > 0.7: Good retrieval quality
- MRR > 0.8: Excellent retrieval quality

#### 3. Modality Gap (Maximum Mean Discrepancy)
**Definition**: Distribution distance between modality embeddings

**Formula**: Uses RBF kernel to measure distribution shift

**Interpretation**:
- Gap < 0.1: Well-aligned modalities
- Gap > 0.3: Significant modality shift

### Text Embedding Metrics

#### 1. Spearman Correlation (STS)
**Definition**: Rank correlation between predicted and ground-truth similarities

**Range**: [-1, 1] where 1 = perfect correlation

**Interpretation**:
- ρ > 0.8: Strong similarity understanding
- ρ > 0.9: State-of-the-art performance

#### 2. V-Measure (Clustering)
**Definition**: Harmonic mean of homogeneity and completeness

**Formula**:
```
V = 2 × (homogeneity × completeness) / (homogeneity + completeness)
```

**Interpretation**:
- V > 0.6: Good clustering
- V > 0.7: Excellent clustering

#### 3. MAP (Mean Average Precision)
**Definition**: Average precision across all queries

**Interpretation**:
- MAP > 0.5: Good retrieval
- MAP > 0.7: Strong retrieval
- MAP > 0.8: State-of-the-art

### Decoder Metrics

#### 1. BLEU (Bilingual Evaluation Understudy)
**Definition**: N-gram precision between generated and reference text

**Variants**:
- BLEU-1: Unigram precision
- BLEU-4: Up to 4-gram precision

**Interpretation**:
- BLEU-4 > 0.3: Decent reconstruction
- BLEU-4 > 0.4: Good reconstruction
- BLEU-4 > 0.5: Excellent reconstruction

#### 2. Semantic Fidelity
**Definition**: Cosine similarity after encode→decode→encode cycle

**Formula**:
```
Fidelity = cos_sim(original_vector, reconstructed_vector)
```

**Interpretation**:
- Fidelity > 0.85: Semantic meaning preserved
- Fidelity > 0.95: Near-perfect preservation

## Running Evaluations

### Quick Evaluation

```bash
# Run comprehensive evaluation
python scripts/run_full_evaluation.py

# Evaluate specific encoders
python scripts/run_evaluation.py --encoders text audio --split test

# Generate performance figures
python scripts/generate_performance_figures.py --publication --dpi 300
```

### MTEB Evaluation

```bash
# Quick MTEB evaluation (5 diverse tasks)
python -c "
from src.Encoders import Text_to_Vec
from src.Evaluation.benchmarks.mteb_adapter import run_mteb_evaluation
encoder = Text_to_Vec()
results = run_mteb_evaluation(encoder, quick_eval=True)
"

# Full MTEB evaluation (warning: takes hours)
python -c "
from src.Encoders import Text_to_Vec
from src.Evaluation.benchmarks.mteb_adapter import run_mteb_evaluation
encoder = Text_to_Vec()
results = run_mteb_evaluation(encoder, quick_eval=False)
"

# Specific category evaluation
python -c "
from src.Encoders import Text_to_Vec
from src.Evaluation.benchmarks.mteb_adapter import MTEBEvaluator
encoder = Text_to_Vec()
evaluator = MTEBEvaluator(encoder)
results = evaluator.run_category('STS')
"
```

### GLUE Evaluation

```bash
# Evaluate on specific GLUE tasks
python -c "
from src.Encoders import Text_to_Vec
from src.Evaluation.benchmarks.glue_adapter import run_glue_evaluation
encoder = Text_to_Vec()
results = run_glue_evaluation(encoder, tasks=['sst2', 'mrpc', 'stsb'])
"

# Full GLUE evaluation
python -c "
from src.Encoders import Text_to_Vec
from src.Evaluation.benchmarks.glue_adapter import GLUEEvaluator
encoder = Text_to_Vec()
evaluator = GLUEEvaluator(encoder)
results = evaluator.run_all()
"
```

### Cross-Modal Evaluation

```bash
# Evaluate cross-modal retrieval
python scripts/run_evaluation.py \
    --split test \
    --k-values 1 5 10 \
    --save-embeddings

# With visualization
python scripts/run_full_evaluation.py \
    --clustering-method tsne \
    --publication-quality
```

## Interpreting Results

### Performance Tiers

#### Tier 1: State-of-the-Art (SOTA)
- Cross-modal R@1 > 0.7
- MTEB average > 0.75
- GLUE average > 0.85
- Semantic fidelity > 0.95

#### Tier 2: Strong Performance
- Cross-modal R@1 > 0.6
- MTEB average > 0.65
- GLUE average > 0.75
- Semantic fidelity > 0.85

#### Tier 3: Good Performance
- Cross-modal R@1 > 0.5
- MTEB average > 0.55
- GLUE average > 0.65
- Semantic fidelity > 0.75

#### Tier 4: Baseline
- Cross-modal R@1 > 0.3
- MTEB average > 0.45
- GLUE average > 0.55
- Semantic fidelity > 0.65

### Key Indicators

**Good Alignment**:
- Low modality gap (< 0.1)
- High cross-modal retrieval (R@10 > 0.9)
- Consistent performance across tasks
- Low variance in semantic fidelity

**Problems to Watch**:
- High modality gap (> 0.3) → Modalities not well aligned
- Low R@1 (< 0.3) → Poor semantic matching
- High BLEU but low fidelity → Surface-level reconstruction
- Task-specific drops → Overfitting or underfitting

### Comparison with Baselines

| Model | Parameters | MTEB Avg | GLUE Avg | Speed | Our Advantage |
|-------|------------|----------|----------|-------|---------------|
| all-MiniLM-L6-v2 | 22M | 63.0 | 77.0 | Fast | Better cross-modal |
| all-mpnet-base-v2 | 109M | 69.7 | 82.0 | Medium | Multimodal support |
| e5-large-v2 | 335M | 75.4 | 85.0 | Slow | Lighter adapters |
| text-embedding-ada-002 | Unknown | 76.0 | 87.0 | API | Open source |

**Our Architecture Advantages**:
1. **Multimodal**: Handles text, audio, and image
2. **Efficient**: Only trains small adapters
3. **Modular**: Mix and match encoders/decoders
4. **Extensible**: Easy to add new modalities

## Performance Baselines

### Expected Performance by Training Stage

#### Stage 1: Initial Training (1-5 epochs)
- Cross-modal R@1: 0.3-0.4
- MTEB average: 0.45-0.55
- GLUE average: 0.60-0.70
- Training loss: > 0.5

#### Stage 2: Mid Training (5-20 epochs)
- Cross-modal R@1: 0.5-0.6
- MTEB average: 0.55-0.65
- GLUE average: 0.70-0.75
- Training loss: 0.2-0.5

#### Stage 3: Converged (20+ epochs)
- Cross-modal R@1: 0.65-0.75
- MTEB average: 0.65-0.75
- GLUE average: 0.75-0.85
- Training loss: < 0.2

### Performance by Data Source

#### MOSI Only
- Best for: Cross-modal alignment
- Limitation: Limited text diversity
- Expected MTEB: 0.50-0.60

#### MTEB + GLUE
- Best for: Text understanding
- Limitation: No multimodal data
- Expected cross-modal R@1: 0.40-0.50

#### Combined (MOSI + MTEB + GLUE)
- Best for: Balanced performance
- Trade-off: Longer training time
- Expected overall: 0.65-0.75

## Troubleshooting

### Common Issues and Solutions

#### 1. Low Cross-Modal Retrieval
**Symptoms**: R@1 < 0.3, R@10 < 0.6
**Solutions**:
- Increase contrastive temperature (try 0.05-0.1)
- Use harder negatives in training
- Increase batch size for more negative samples
- Check data augmentation pipeline

#### 2. Poor MTEB Performance
**Symptoms**: MTEB avg < 0.5
**Solutions**:
- Add more MTEB training data
- Fine-tune on specific task categories
- Adjust adapter learning rate
- Check text preprocessing

#### 3. Modality Gap Too Large
**Symptoms**: MMD > 0.3
**Solutions**:
- Increase momentum for momentum encoder
- Use cross-modal mixup augmentation
- Add alignment regularization term
- Balance sampling across modalities

#### 4. Decoder Semantic Drift
**Symptoms**: Fidelity < 0.7
**Solutions**:
- Increase weight of semantic loss
- Use cycle consistency loss
- Reduce decoder learning rate
- Add skip connections

#### 5. Training Instability
**Symptoms**: Loss spikes, NaN values
**Solutions**:
- Reduce learning rate
- Enable gradient clipping
- Check for data loading errors
- Use mixed precision carefully

### Debugging Commands

```bash
# Check encoder outputs
python -c "
import torch
from src.Encoders import Text_to_Vec
encoder = Text_to_Vec()
output = encoder(['test text'])
print(f'Output shape: {output.shape}')
print(f'Output norm: {torch.norm(output)}')
print(f'Contains NaN: {torch.isnan(output).any()}')
"

# Validate dataset
python -c "
from src.Training.Data_Wrangling.benchmark_dataset import BenchmarkDataset
dataset = BenchmarkDataset(data_sources=['mteb', 'glue'])
print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
"

# Quick metric check
python -c "
import torch
from src.Evaluation.encoder_metrics import compute_recall_at_k
q = torch.randn(10, 1024)
t = torch.randn(10, 1024)
metrics = compute_recall_at_k(q, t)
print(f'Random baseline R@1: {metrics[\"R@1\"]:.3f}')
"
```

## Best Practices

### For Research Papers
1. Report metrics with confidence intervals
2. Use at least 3 random seeds
3. Include both peak and final performance
4. Report computational cost (FLOPs, time)
5. Compare with relevant baselines

### For Production
1. Optimize for target metric (not average)
2. Use model distillation for speed
3. Cache embeddings when possible
4. Monitor drift in production
5. A/B test against baseline

### For Development
1. Start with quick evaluation subset
2. Use tensorboard for monitoring
3. Save checkpoints frequently
4. Version control configurations
5. Document hyperparameter choices

## References

### Papers
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)
- [GLUE: A Multi-Task Benchmark](https://arxiv.org/abs/1804.07461)
- [MultiBench: Multimodal Learning Benchmark](https://arxiv.org/abs/2107.07502)
- [CMU-MOSI: Multimodal Corpus](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)

### Leaderboards
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [GLUE Leaderboard](https://gluebenchmark.com/leaderboard)
- [Papers with Code - Text Embeddings](https://paperswithcode.com/task/text-embeddings)

### Tools
- [HuggingFace Evaluate](https://github.com/huggingface/evaluate)
- [Sentence Transformers](https://www.sbert.net/)
- [CMU MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)

---

*Last updated: December 2024*
*For questions or improvements, please open an issue on GitHub*