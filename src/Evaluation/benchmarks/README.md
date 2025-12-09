# Benchmark Adapters

Adapter modules for running standard NLP and multimodal benchmarks with our encoder architecture.

## Available Adapters

| Adapter | File | Benchmark | Tasks |
|---------|------|-----------|-------|
| MTEBEvaluator | `mteb_adapter.py` | MTEB | 56+ text tasks |
| GLUEEvaluator | `glue_adapter.py` | GLUE | 9 NLU tasks |
| MultiBenchEvaluator | `multibench_adapter.py` | MultiBench | 15 multimodal datasets |

## MTEB Adapter

Evaluates text embeddings on the Massive Text Embedding Benchmark (56+ tasks across 8 categories).

### Quick Start

```python
from src.Encoders import Text_to_Vec
from src.Evaluation.benchmarks.mteb_adapter import MTEBEvaluator

encoder = Text_to_Vec()
evaluator = MTEBEvaluator(encoder)

# Quick evaluation (5 representative tasks)
results = evaluator.run(quick_eval=True)

# Specific tasks
results = evaluator.run(tasks=['STS12', 'STS13', 'Banking77Classification'])

# Full category
results = evaluator.run_category('STS')
```

### Task Categories

| Category | Tasks | Metrics | Description |
|----------|-------|---------|-------------|
| STS | STS12-22, STSBenchmark | Spearman | Semantic similarity |
| Retrieval | MSMARCO, NQ, HotpotQA | nDCG@10 | Information retrieval |
| Clustering | ArXiv, Reddit, StackEx | V-measure | Document clustering |
| Classification | Banking77, Amazon | Accuracy | Text categorization |
| PairClassification | QQP, TwitterSemEval | AP | Pair classification |
| Reranking | AskUbuntu | MAP | Query reranking |
| Bitext Mining | Tatoeba | F1 | Parallel sentences |
| Summarization | SummEval | Spearman | Summarization quality |

### Key Tasks for Our Architecture

```python
# Tasks most relevant to semantic vector space
STS_TASKS = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']
RETRIEVAL_TASKS = ['MSMARCO', 'NaturalQuestions', 'HotpotQA']
CLUSTERING_TASKS = ['ArxivClusteringP2P', 'RedditClustering']

evaluator.run(tasks=STS_TASKS + RETRIEVAL_TASKS)
```

### Helper Functions

```python
from src.Evaluation.benchmarks.mteb_adapter import run_mteb_evaluation

# Simple wrapper
results = run_mteb_evaluation(encoder, quick_eval=True)
print(f"Average score: {results['average']:.3f}")
```

## GLUE Adapter

Evaluates on the General Language Understanding Evaluation benchmark (9 NLU tasks).

### Quick Start

```python
from src.Encoders import Text_to_Vec
from src.Evaluation.benchmarks.glue_adapter import GLUEEvaluator

encoder = Text_to_Vec()
evaluator = GLUEEvaluator(encoder)

# Specific tasks
results = evaluator.run(tasks=['sst2', 'mrpc', 'stsb'])

# All tasks
results = evaluator.run_all()
```

### Task Reference

| Task | Type | Size | Metric | Description |
|------|------|------|--------|-------------|
| CoLA | Single | 10.7k | Matthews | Grammatical acceptability |
| SST-2 | Single | 70k | Accuracy | Sentiment analysis |
| MRPC | Pair | 5.8k | F1/Acc | Paraphrase detection |
| QQP | Pair | 404k | F1/Acc | Question duplicates |
| STS-B | Pair | 8.6k | Pearson | Semantic similarity |
| MNLI | Pair | 433k | Accuracy | Natural language inference |
| QNLI | Pair | 115k | Accuracy | Question-answering NLI |
| RTE | Pair | 3k | Accuracy | Textual entailment |
| WNLI | Pair | 852 | Accuracy | Coreference |

### Extract Training Triplets

```bash
# Generate triplets for contrastive training
python scripts/data_wrangling/wrangle_glue_data.py

# Output: data/glue/glue_triplets.pkl
```

### Helper Functions

```python
from src.Evaluation.benchmarks.glue_adapter import run_glue_evaluation

results = run_glue_evaluation(encoder, tasks=['sst2', 'mrpc'])
print(f"SST-2 accuracy: {results['sst2']['accuracy']:.3f}")
```

## MultiBench Adapter

Evaluates on the MultiBench multimodal learning benchmark (15 datasets across 10 modalities).

### Quick Start

```python
from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from src.Evaluation.benchmarks.multibench_adapter import MultiBenchEvaluator

encoders = {
    'text': Text_to_Vec(),
    'audio': Audio_to_Vec(),
    'image': Image_to_Vec()
}
evaluator = MultiBenchEvaluator(encoders)

# Run specific tasks
results = evaluator.run(tasks=['mosi_sentiment', 'mosei_sentiment'])
```

### Key Datasets

| Dataset | Modalities | Task | Metrics |
|---------|------------|------|---------|
| MOSI | Text, Audio, Video | Sentiment | MAE, Corr, Acc |
| MOSEI | Text, Audio, Video | Sentiment | MAE, Corr, Acc |
| UR-FUNNY | Text, Audio, Video | Humor detection | Accuracy |
| AV-MNIST | Audio, Image | Digit recognition | Accuracy |
| Kinetics-Sounds | Audio, Video | Action recognition | Accuracy |

### Multimodal Evaluation

```python
# Evaluate cross-modal alignment
results = evaluator.evaluate_alignment(
    modality_pairs=[('text', 'audio'), ('text', 'image')]
)

# Evaluate fusion quality
results = evaluator.evaluate_fusion(
    fusion_method='concat',  # or 'attention', 'tensor'
    task='mosi_sentiment'
)
```

## Running All Benchmarks

### Quick Evaluation Script

```bash
# Run comprehensive benchmark evaluation
python scripts/evaluation/run_full_evaluation.py

# Specific encoder evaluation
python scripts/evaluation/run_evaluation.py --encoders text audio --split test
```

### Programmatic Usage

```python
from src.Evaluation.benchmarks import MTEBEvaluator, GLUEEvaluator, MultiBenchEvaluator

# Initialize
encoder = Text_to_Vec()

# Run all benchmarks
mteb_results = MTEBEvaluator(encoder).run(quick_eval=True)
glue_results = GLUEEvaluator(encoder).run(['sst2', 'mrpc', 'stsb'])

print(f"MTEB average: {mteb_results['average']:.3f}")
print(f"GLUE average: {sum(glue_results.values()) / len(glue_results):.3f}")
```

## Performance Expectations

### After Token-Based Training

| Benchmark | Metric | Expected Range |
|-----------|--------|----------------|
| MTEB (STS) | Spearman | 0.65-0.75 |
| MTEB (Retrieval) | nDCG@10 | 0.50-0.65 |
| GLUE (SST-2) | Accuracy | 0.85-0.90 |
| GLUE (STS-B) | Pearson | 0.80-0.88 |
| MultiBench (MOSI) | Accuracy | 0.75-0.82 |

### Baseline Comparisons

| Model | MTEB Avg | GLUE Avg | Notes |
|-------|----------|----------|-------|
| all-MiniLM-L6-v2 | 63.0 | 77.0 | Fast, small |
| all-mpnet-base-v2 | 69.7 | 82.0 | Medium size |
| e5-large-v2 | 75.4 | 85.0 | Large model |
| **Our approach** | 65-70 | 78-83 | Multimodal + efficient |

## Troubleshooting

### Low MTEB Scores

```python
# Check encoder output normalization
output = encoder(["test"])
print(f"Norm: {torch.norm(output)}")  # Should be ~1.0

# Verify pooling strategy
# Some tasks prefer mean pooling vs CLS token
```

### GLUE Task Failures

```python
# Verify task-specific preprocessing
from src.Evaluation.benchmarks.glue_adapter import preprocess_for_task
processed = preprocess_for_task(text, task='mrpc')
```

### MultiBench Memory Issues

```python
# Reduce batch size
evaluator.run(tasks=['mosi_sentiment'], batch_size=16)

# Use CPU for evaluation
evaluator.run(tasks=['mosi_sentiment'], device='cpu')
```

## Related Documentation

- [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md) - Benchmark overview and interpretation
- [docs/EVALUATION.md](../../../docs/EVALUATION.md) - Evaluation philosophy
- [src/Evaluation/README.md](../README.md) - Full evaluation API
