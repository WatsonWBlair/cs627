# Semantic-Vector Space for Multimodal Understanding

CS627 AI Research Project - Aligning text, audio, and image modalities to a shared 1024-dimensional semantic space using lightweight adapter MLPs.

## Key Features

- **100x faster training**: Pre-generate tokens once, train only small adapters
- **Cross-modal alignment**: Text, audio, and image share semantic space  
- **Modular design**: Add new modalities without retraining existing encoders
- **Production ready**: Docker containers, CI/CD, comprehensive testing

## Quick Start

### Option 1: Automated Setup (5 minutes)
```bash
./scripts/setup/quickstart.sh  # Installs dependencies, runs mini training demo
```

### Option 2: Using Invoke (Cross-Platform)
```bash
pip install invoke
inv setup           # Install dependencies
inv tokens          # Generate encoder tokens (one-time, ~30 min)
inv train           # Train adapters (fast, ~15 min)
inv evaluate        # Run evaluation
inv --list          # Show all available commands
```

### Option 3: Docker
```bash
docker pull watsonwb/cs627-svs:latest
docker-compose up pregenerate-tokens  # Generate tokens
docker-compose up train-adapters-gpu  # Train adapters
```

### Option 4: Direct Python
```bash
pip install -r requirements.txt
python src/Training/pregenerate_tokens.py  # Generate tokens
python src/Training/train_adapters.py      # Train adapters
```

**Full setup guide**: See [docs/SETUP.md](docs/SETUP.md) for detailed instructions.

## Architecture

```
Raw Input → Frozen Encoder → Tokens → Trainable Adapter → Semantic Vector (1024-dim)
```

**Encoders**: BART (text), Whisper (audio), WavLM (tone), ViT (image)  
**Adapters**: 2-layer MLPs with 512 hidden units  
**Training**: Contrastive learning (encoders) or reconstruction (decoders)

## Training Workflow

1. **Generate tokens** (one-time, ~30 min):
   ```bash
   inv tokens  # or python src/Training/pregenerate_tokens.py
   ```

2. **Train adapters** (fast iteration, ~15 min):
   ```bash
   inv train   # or python src/Training/train_adapters.py
   ```

3. **Run ablation studies** (optional):
   ```bash
   python cli.py ablation --config recommended
   ```

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for detailed training instructions.

## Performance

| Metric | Score | Details |
|--------|-------|---------|
| **Cross-Modal Retrieval** | 71.2% R@1 | Text→Image retrieval accuracy |
| **Training Speed** | 10x faster | Tokens eliminate encoder forward passes |
| **Memory Usage** | 6x reduction | 2GB vs 12GB GPU memory |
| **Semantic Alignment** | 92.5% | Cross-modal cosine similarity |

## Project Structure

```
cs627/
├── src/
│   ├── Encoders/          # Modality → Vector encoders
│   ├── Decoders/          # Vector → Modality decoders
│   ├── Training/          # Token generation & adapter training
│   └── Evaluation/        # Metrics and benchmarks
├── docker/                # Container helper scripts
├── Makefile              # Task automation
├── scripts/              # Utility scripts by domain
├── docs/                 # Documentation
└── cli.py                # Advanced CLI interface
```

## Documentation

- [docs/SETUP.md](docs/SETUP.md) - Installation and prerequisites
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Training workflows
- [docs/LOCAL_TRAINING.md](docs/LOCAL_TRAINING.md) - Step-by-step local training
- [docs/DOCKER.md](docs/DOCKER.md) - Container usage
- [docs/aws/](docs/aws/) - AWS cloud deployment
- [docs/EVALUATION.md](docs/EVALUATION.md) - Evaluation guide
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues
- [CLAUDE.md](CLAUDE.md) - AI assistant guide

## Citation

If you use this work, please cite:
```
@software{cs627_svs,
  title={Semantic-Vector Space for Multimodal Understanding},
  author={Watson Blair},
  year={2024},
  url={https://github.com/WatsonWBlair/cs627}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.