# Experiments Module

Tools for running ablation studies and systematic architecture experiments on adapter configurations.

**Related**: See [docs/TRAINING_GUIDE.md](../../docs/TRAINING_GUIDE.md) for training concepts and best practices.

## Available Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `run_ablation.py` | Run ablation studies | Results JSON/CSV |
| `adapter_configs.py` | Generate config sets | Config JSON files |

## Quick Start

```bash
# Run recommended configurations (3 configs, ~15 min)
python src/Experiments/run_ablation.py --config recommended

# Run full grid search (192 configs, ~8 hours)
python src/Experiments/run_ablation.py --config grid_search

# Run random search (20 configs, ~1 hour)
python src/Experiments/run_ablation.py --config random_search
```

## Configuration Sets

| Config Set | Configs | Purpose | Est. Time |
|------------|---------|---------|-----------|
| `recommended` | 3 | Best known configurations | ~15 min |
| `grid_search` | 192 | Exhaustive hyperparameter sweep | ~8 hours |
| `random_search` | 20 | Quick exploration | ~1 hour |
| `progressive_depth` | 4 | Study layer depth impact | ~20 min |
| `activation_study` | 3 | Compare activation functions | ~15 min |
| `dropout_study` | 4 | Study dropout rates | ~20 min |

## Hyperparameter Ranges

### Default Grid Search Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| `hidden_size` | 128, 256, 512, 1024 | MLP hidden units |
| `num_layers` | 1, 2, 3, 4 | Number of hidden layers |
| `activation` | relu, gelu, silu | Activation function |
| `dropout` | 0.0, 0.1, 0.2, 0.3 | Dropout rate |

### Recommended Configurations

Based on preliminary experiments:

```python
RECOMMENDED_CONFIGS = [
    {'hidden_size': 512, 'num_layers': 2, 'activation': 'relu', 'dropout': 0.1},
    {'hidden_size': 256, 'num_layers': 3, 'activation': 'gelu', 'dropout': 0.1},
    {'hidden_size': 1024, 'num_layers': 1, 'activation': 'relu', 'dropout': 0.0},
]
```

## Usage

### Run Pre-configured Studies

```bash
# Recommended configs (fastest)
python src/Experiments/run_ablation.py --config recommended

# Compare activation functions
python src/Experiments/run_ablation.py --config activation_study

# Study dropout impact
python src/Experiments/run_ablation.py --config dropout_study

# Study depth impact
python src/Experiments/run_ablation.py --config progressive_depth
```

### Custom Configuration

Create `configs/ablation/custom.json`:

```json
{
  "num_configs": 3,
  "configs": [
    {
      "name": "small_fast",
      "hidden_size": 256,
      "num_layers": 1,
      "activation": "relu",
      "dropout": 0.0
    },
    {
      "name": "medium_balanced",
      "hidden_size": 512,
      "num_layers": 2,
      "activation": "gelu",
      "dropout": 0.1
    },
    {
      "name": "large_capacity",
      "hidden_size": 1024,
      "num_layers": 3,
      "activation": "silu",
      "dropout": 0.2
    }
  ]
}
```

Run custom study:

```bash
python src/Experiments/run_ablation.py --config-file configs/ablation/custom.json
```

### Command Line Options

```bash
python src/Experiments/run_ablation.py \
    --config recommended \       # Config set to use
    --epochs 30 \                # Epochs per configuration
    --lr 0.001 \                 # Learning rate
    --token-dir data/pregenerated_tokens/mosi  # Token directory
```

## Output Format

### Results Directory Structure

```
Results/ablation/
├── ablation_results_YYYYMMDD_HHMMSS.json  # Full results
└── ablation_summary_YYYYMMDD_HHMMSS.csv   # Summary table
```

### JSON Results Format

```json
{
  "config": {
    "name": "adapter_0",
    "hidden_size": 512,
    "num_layers": 2,
    "activation": "relu",
    "dropout": 0.1
  },
  "identifier": "h512_l2_relu_d1",
  "epochs_trained": 30,
  "best_val_loss": 0.342,
  "best_metrics": {...},
  "final_recall_at_1": 0.523,
  "final_recall_at_5": 0.847,
  "total_parameters": 1935360,
  "model_size_mb": 7.38,
  "history": {
    "train_loss": [...],
    "val_loss": [...],
    "val_recall_at_1": [...],
    "val_recall_at_5": [...]
  }
}
```

### CSV Summary Columns

| Column | Description |
|--------|-------------|
| `name` | Configuration name |
| `identifier` | Unique config ID (e.g., `h512_l2_relu_d1`) |
| `hidden_size` | MLP hidden units |
| `num_layers` | Number of layers |
| `activation` | Activation function |
| `dropout` | Dropout rate |
| `best_val_loss` | Best validation loss |
| `final_recall_at_1` | Final R@1 metric |
| `final_recall_at_5` | Final R@5 metric |
| `epochs_trained` | Epochs before early stopping |
| `total_parameters` | Total trainable parameters |
| `model_size_mb` | Model size in MB |

## Analysis

### Automatic Analysis

The runner automatically computes:

1. **Best configuration** - Lowest validation loss
2. **Most efficient** - Best performance/size ratio
3. **Parameter correlations** - How each hyperparameter affects loss
4. **Per-parameter impact** - Mean/std/min for each parameter value

### Manual Analysis

```python
import pandas as pd
import json

# Load results
df = pd.read_csv('Results/ablation/ablation_summary_YYYYMMDD_HHMMSS.csv')

# Best by validation loss
print(df.nsmallest(5, 'best_val_loss'))

# Best by recall
print(df.nlargest(5, 'final_recall_at_1'))

# Impact of hidden size
print(df.groupby('hidden_size')['best_val_loss'].agg(['mean', 'std', 'min']))

# Impact of activation
print(df.groupby('activation')['final_recall_at_1'].agg(['mean', 'std', 'max']))
```

## Programmatic Usage

### Generate Configurations

```python
from src.Experiments.adapter_configs import ConfigGenerator, AdapterConfig

generator = ConfigGenerator()

# Generate different sets
grid_configs = generator.generate_grid_search()
random_configs = generator.generate_random_search(20)
recommended = generator.generate_recommended()

# Save to file
generator.save_configs(grid_configs, 'configs/ablation/my_configs.json')

# Load from file
configs = ConfigGenerator.load_configs('configs/ablation/my_configs.json')
```

### Run Custom Ablation

```python
from src.Experiments.run_ablation import AblationRunner
from src.Experiments.adapter_configs import ConfigGenerator

# Initialize runner
runner = AblationRunner(
    token_dir='data/pregenerated_tokens/mosi',
    results_dir='Results/ablation'
)

# Generate configs
generator = ConfigGenerator()
configs = generator.generate_recommended()

# Run ablation
results_df = runner.run_ablation(
    configs,
    epochs_per_config=30,
    learning_rate=0.001
)

# Analyze results
analysis = runner.analyze_results(results_df)
print(f"Best config: {analysis['best_config']['identifier']}")
```

### Estimate Training Time

```python
from src.Experiments.adapter_configs import ConfigGenerator, estimate_training_time

generator = ConfigGenerator()
configs = generator.generate_grid_search()

estimates = estimate_training_time(configs, epochs_per_config=30)
print(f"Estimated time: {estimates['total_hours']:.1f} hours")
print(f"Per config: {estimates['average_minutes_per_config']:.1f} minutes")
```

## Prerequisites

Before running experiments:

1. **Pre-generate tokens** (required):
   ```bash
   python src/Training/pregenerate_tokens.py
   ```

2. **Verify tokens exist**:
   ```bash
   ls -lh data/pregenerated_tokens/mosi/
   # Should see: train_tokens.h5, val_tokens.h5, test_tokens.h5
   ```

## Docker Support

For containerized experiments, use the ablation image:

```bash
# Build ablation-specific image
docker build -f docker/Dockerfile.ablation -t cs627-ablation .

# Run ablation study
docker run --gpus all -v $(pwd)/Results:/app/Results cs627-ablation \
    python src/Experiments/run_ablation.py --config recommended
```

See [docs/DOCKER.md](../../docs/DOCKER.md) for container details.

## Related Documentation

- [docs/TRAINING_GUIDE.md](../../docs/TRAINING_GUIDE.md) - Training workflow and best practices
- [src/Training/README.md](../Training/README.md) - Adapter training implementation
- [docs/EVALUATION.md](../../docs/EVALUATION.md) - Evaluation metrics and methodology
