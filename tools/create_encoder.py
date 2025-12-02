#!/usr/bin/env python3
"""
Encoder Generator Script

Creates a new encoder module from the boilerplate template with user-specified parameters.

Usage:
    python tools/create_encoder.py --modality audio --model openai/whisper-small --output-dim 1024
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def generate_encoder(
    modality: str,
    base_model: str,
    output_dim: int = 1024,
    hidden_size: int = 200,
    hidden_layers: int = 2,
    freeze_encoder: bool = True,
    experimental: bool = True,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate encoder module from boilerplate template.

    Args:
        modality: Modality name (e.g., 'audio', 'video', 'text')
        base_model: HuggingFace model identifier
        output_dim: Semantic vector dimension
        hidden_size: Adapter hidden layer size
        hidden_layers: Number of adapter hidden layers
        freeze_encoder: Whether to freeze pretrained encoder
        output_path: Custom output path (default: src/Encoders/{modality}_2_vec.py)

    Returns:
        Path to generated encoder file
    """
    # Read boilerplate template
    boilerplate_path = Path("src/Encoders/encoder_boilerplate.py")
    if not boilerplate_path.exists():
        print(f"Error: Boilerplate not found at {boilerplate_path}")
        sys.exit(1)

    with open(boilerplate_path, 'r') as f:
        template = f.read()

    # Generate class name
    class_name = f"{modality.capitalize()}_to_Vec"

    # Replace placeholders
    model_safe = base_model.replace('/', '_')

    replacements = {
        '{Modality}': modality.capitalize(),
        'TODO_BASE_MODEL': base_model,
        'TODO_PROCESSOR_CLASS': 'AutoProcessor',  # User needs to update this
        'TODO_MODEL_CLASS': 'AutoModel',  # User needs to update this
        'TODO_ENCODER_DIM': '768',  # User needs to update this
        '{modality}': modality.lower(),
        '1024': str(output_dim),
        '200': str(hidden_size),
        '2': str(hidden_layers),
        'True': str(freeze_encoder)
    }

    content = template
    for old, new in replacements.items():
        content = content.replace(old, new)

    # Add experimental marker if requested
    if experimental:
        experimental_marker = '''"""
EXPERIMENTAL: This encoder is not fully implemented.

Status: Under Development
Issues:
- Adapter integration needs validation
- Training loop not tested
- Cross-modal alignment not verified

DO NOT USE IN PRODUCTION.
"""

'''
        content = experimental_marker + content

    # Determine output path
    if output_path is None:
        output_path = Path(f"src/Encoders/{modality.lower()}_2_vec.py")

    # Check if file already exists
    if output_path.exists():
        response = input(f"File {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Write generated encoder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

    print(f"✓ Generated encoder: {output_path}")
    print(f"✓ Class name: {class_name}")
    print(f"✓ Base model: {base_model}")
    if experimental:
        print("✓ Marked as EXPERIMENTAL")
    print()
    print("Next steps:")
    print(f"  1. Edit {output_path} and replace TODO markers:")
    print("     - TODO_PROCESSOR_CLASS (e.g., WhisperProcessor, ViTImageProcessor)")
    print("     - TODO_MODEL_CLASS (e.g., WhisperForConditionalGeneration)")
    print("     - TODO_ENCODER_DIM (e.g., 768 for Whisper, 1024 for ViT)")
    print(f"  2. Test the encoder:")
    print(f"     python -c \"from src.Encoders.{modality.lower()}_2_vec import {class_name}; enc = {class_name}()\"")
    print("  3. Train the encoder adapter:")
    print("     python src/Training/train_encoder_alignment.py")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new encoder module from boilerplate template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create audio encoder with Whisper
  python tools/create_encoder.py --modality audio --model openai/whisper-small

  # Create video encoder with custom dimensions
  python tools/create_encoder.py --modality video --model microsoft/xclip-base-patch32 --output-dim 512

  # Create encoder with custom adapter architecture
  python tools/create_encoder.py --modality depth --model intel/dpt-large --hidden-size 512 --hidden-layers 3
        """
    )

    parser.add_argument(
        '--modality',
        type=str,
        required=True,
        help='Modality name (e.g., audio, video, depth, radar)'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace model identifier (e.g., openai/whisper-small)'
    )
    parser.add_argument(
        '--output-dim',
        type=int,
        default=1024,
        help='Semantic vector dimension (default: 1024)'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=200,
        help='Adapter hidden layer size (default: 200)'
    )
    parser.add_argument(
        '--hidden-layers',
        type=int,
        default=2,
        help='Number of adapter hidden layers (default: 2)'
    )
    parser.add_argument(
        '--no-freeze',
        action='store_true',
        help='Do not freeze pretrained encoder (train full model)'
    )
    parser.add_argument(
        '--no-experimental',
        action='store_true',
        help='Do not mark encoder as experimental'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Custom output path (default: src/Encoders/{modality}_2_vec.py)'
    )

    args = parser.parse_args()

    # Validate modality name
    if not args.modality.isalnum():
        print(f"Error: Modality name must be alphanumeric, got: {args.modality}")
        sys.exit(1)

    # Generate encoder
    output_path = generate_encoder(
        modality=args.modality,
        base_model=args.model,
        output_dim=args.output_dim,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        freeze_encoder=not args.no_freeze,
        experimental=not args.no_experimental,
        output_path=args.output
    )

    print()
    print("✓ Encoder generation complete!")


if __name__ == "__main__":
    main()
