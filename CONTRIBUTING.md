# Contributing Guide

Thank you for contributing to the Semantic-Vector Space project! This guide will help you add new encoders, decoders, and improvements to the codebase.

## Getting Started

1. **Read the documentation**:
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture patterns
   - [EVALUATION.md](EVALUATION.md) - Testing and evaluation
   - [src/Encoders/README.md](src/Encoders/README.md) - Encoder guide
   - [src/Decoders/README.md](src/Decoders/README.md) - Decoder guide

2. **Set up your environment**:
   ```bash
   git clone https://github.com/WatsonWBlair/cs627.git
   cd cs627
   pip install -r requirements.txt
   ```

## Adding a New Encoder

### Using the Generator (Recommended)

```bash
python tools/create_encoder.py --modality audio --model openai/whisper-small
```

This creates a template encoder file marked as **EXPERIMENTAL** by default. Then:

1. Edit the generated file and replace TODO markers
2. Test the encoder imports and instantiates
3. Train the adapter using contrastive learning
4. Evaluate cross-modal alignment
5. Remove EXPERIMENTAL marker once validated

### Manual Creation

1. Copy `src/Encoders/encoder_boilerplate.py` to `src/Encoders/{modality}_2_vec.py`
2. Add EXPERIMENTAL marker to module docstring
3. Follow the patterns in existing encoders (text_2_vec.py, wav_2_vec.py)
4. Key requirements:
   - Inherit from `torch.nn.Module`
   - Initialize an `Adapter` in `__init__`
   - Implement `forward()` method that returns semantic vector

## Adding a New Decoder

### Using the Generator (Recommended)

```bash
python tools/create_decoder.py --modality audio --model suno/bark-small
```

**Important**: An encoder for the same modality must exist before creating a decoder.

Then:

1. Edit the generated file and replace TODO markers
2. Implement the generation logic in `forward()`
3. Test the decoder imports and instantiates
4. Train using dual-loss backtranslation

### Manual Creation

1. Copy `src/Decoders/decoder_boilerplate.py` to `src/Decoders/vec_2_{modality}.py`
2. Add EXPERIMENTAL marker to module docstring (if not already present)
3. Follow the patterns in existing decoders (vec_2_text.py)
4. Key requirements:
   - Inherit from `torch.nn.Module`
   - Initialize an `Adapter` in `__init__`
   - Implement `forward()` method that generates output

## Marking Modules as Production-Ready

New encoders and decoders should be marked **EXPERIMENTAL** until fully tested. To mark a module as production-ready:

1. Remove the EXPERIMENTAL marker from the module docstring
2. Ensure the module passes functional tests:
   ```bash
   # For encoders
   python tools/validate_module.py src/Encoders/{modality}_2_vec.py

   # For decoders
   python tools/validate_module.py src/Decoders/vec_2_{modality}.py
   ```
3. Complete evaluation using metrics from EVALUATION.md
4. Document evaluation results in a PR

## Testing Your Contribution

### Functional Testing

Experimental modules are skipped. Only production-ready modules are functionally tested.

```bash
# Test single module
python tools/validate_module.py src/Encoders/{modality}_2_vec.py
python tools/validate_module.py src/Decoders/vec_2_{modality}.py

# Test all modules
python tools/validate_module.py --all
```

### Manual Testing

```python
# Test encoder
from src.Encoders.{modality}_2_vec import {Modality}_to_Vec
encoder = {Modality}_to_Vec()
vector = encoder(sample_data)
print(f"Output shape: {vector.shape}")  # Should be (batch_size, 1024)

# Test decoder
from src.Decoders.vec_2_{modality} import Vec_to_{Modality}
decoder = Vec_to_{Modality}()
output = decoder(vector)
```

## Architecture Guidelines

### Naming Conventions

- **Encoders**: `{Modality}_to_Vec` (e.g., `Audio_to_Vec`)
- **Decoders**: `Vec_to_{Modality}` (e.g., `Vec_to_Audio`)
- **Files**: `{modality}_2_vec.py` for encoders, `vec_2_{modality}.py` for decoders

### Required Components

**All modules must**:
- Inherit from `torch.nn.Module`
- Initialize an `Adapter` instance
- Implement a `forward()` method
- Include type hints (recommended)

**Adapters should**:
- Use prefix naming: `{model_name}_{enc/dec}`
- Default dimensions: input/output 1024, hidden 200, layers 2
- Be saved to `OptimalWeights/{prefix}_weights.pth`

### Code Style

- Follow existing patterns in the codebase
- Add docstrings to classes and methods
- Use type hints where helpful
- Keep code simple and readable

## Submitting Changes

### Creating a Pull Request

1. Create a feature branch:
   ```bash
   git checkout -b feature/add-{modality}-encoder
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add {modality} encoder with {model}"
   ```

3. Push and create a PR:
   ```bash
   git push origin feature/add-{modality}-encoder
   ```

### PR Requirements

**For Production-Ready Modules**:
- [ ] Functional tests pass
- [ ] Evaluation metrics documented
- [ ] EXPERIMENTAL marker removed
- [ ] README updated if needed

**For Experimental Modules**:
- [ ] EXPERIMENTAL marker present
- [ ] Known limitations documented
- [ ] Basic import/instantiation works

### PR Description Template

```markdown
## Summary
[Brief description of changes]

## Type of Change
- [ ] New encoder (Experimental)
- [ ] New encoder (Production-Ready)
- [ ] New decoder (Experimental)
- [ ] New decoder (Production-Ready)
- [ ] Bug fix
- [ ] Documentation
- [ ] Other

## Module Details
- Modality: [text/audio/image/other]
- Base Model: [HuggingFace model ID]
- Status: [Experimental/Production-Ready]
- Classes: [List of encoder/decoder class names]

## Testing
- [ ] Module imports without errors
- [ ] Module instantiates without errors
- [ ] Forward pass works with sample data
- [ ] Evaluation completed (if production-ready)

## Evaluation Results (if production-ready)
[Include key metrics from EVALUATION.md]

## Additional Notes
[Any other relevant information]
```

## Getting Help

- **Architecture questions**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Training questions**: See [src/Training/README.md](src/Training/README.md)
- **Evaluation questions**: See [EVALUATION.md](EVALUATION.md)
- **Issues**: Open an issue on GitHub

## Code of Conduct

- Be respectful and constructive
- Focus on improving the codebase
- Help others learn and contribute
- Follow the contribution guidelines

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Research/Educational Use).
