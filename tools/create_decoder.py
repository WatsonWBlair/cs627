#!/usr/bin/env python3
"""
Decoder Generator Script

Creates a new decoder module from the boilerplate template with user-specified parameters.

Usage:
    python tools/create_decoder.py --modality audio --model suno/bark-small --output-dim 1024
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def check_encoder_exists(modality: str) -> bool:
    """Check if corresponding encoder exists for this modality."""
    encoder_path = Path(f"src/Encoders/{modality.lower()}_2_vec.py")
    return encoder_path.exists()


def generate_decoder(
    modality: str,
    base_model: str,
    output_dim: int = 1024,
    hidden_size: int = 200,
    hidden_layers: int = 2,
    experimental: bool = True,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate decoder module from boilerplate template.

    Args:
        modality: Modality name (e.g., 'audio', 'video', 'text')
        base_model: HuggingFace model identifier
        output_dim: Semantic vector dimension
        hidden_size: Adapter hidden layer size
        hidden_layers: Number of adapter hidden layers
        experimental: Mark decoder as experimental
        output_path: Custom output path (default: src/Decoders/vec_2_{modality}.py)

    Returns:
        Path to generated decoder file
    """
    # Check if encoder exists
    if not check_encoder_exists(modality):
        print(f"Warning: No encoder found for modality '{modality}'")
        print(f"         Expected: src/Encoders/{modality.lower()}_2_vec.py")
        print()
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted. Create encoder first using: python tools/create_encoder.py")
            sys.exit(0)

    # Read boilerplate template
    boilerplate_path = Path("src/Decoders/decoder_boilerplate.py")
    if not boilerplate_path.exists():
        print(f"Error: Boilerplate not found at {boilerplate_path}")
        sys.exit(1)

    with open(boilerplate_path, 'r') as f:
        template = f.read()

    # Generate class name
    class_name = f"Vec_to_{modality.capitalize()}"

    # Replace placeholders
    model_safe = base_model.replace('/', '_')

    replacements = {
        '{Modality}': modality.capitalize(),
        'TODO_BASE_MODEL': base_model,
        'TODO_TOKENIZER_CLASS': 'AutoTokenizer',  # User needs to update this
        'TODO_DECODER_CLASS': 'AutoModelForCausalLM',  # User needs to update this
        '{modality}': modality.lower(),
        '1024': str(output_dim),
        '200': str(hidden_size),
        '2': str(hidden_layers)
    }

    content = template
    for old, new in replacements.items():
        content = content.replace(old, new)

    # Add experimental marker if requested
    if experimental:
        experimental_marker = '''"""
EXPERIMENTAL: This decoder is not fully implemented.

Status: Under Development
Issues:
- Adapter integration needs validation
- Training loop not tested
- Output format needs verification

DO NOT USE IN PRODUCTION.
"""

'''
        content = experimental_marker + content

    # Determine output path
    if output_path is None:
        output_path = Path(f"src/Decoders/vec_2_{modality.lower()}.py")

    # Check if file already exists
    if output_path.exists():
        response = input(f"File {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Write generated decoder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

    print(f"✓ Generated decoder: {output_path}")
    print(f"✓ Class name: {class_name}")
    print(f"✓ Base model: {base_model}")
    if experimental:
        print("✓ Marked as EXPERIMENTAL")
    print()
    print("Next steps:")
    print(f"  1. Edit {output_path} and replace TODO markers:")
    print("     - TODO_TOKENIZER_CLASS (e.g., BartTokenizer, WhisperTokenizer)")
    print("     - TODO_DECODER_CLASS (e.g., BartForConditionalGeneration)")
    print(f"  2. Implement the forward() method for {modality} generation")
    print(f"  3. Test the decoder:")
    print(f"     python -c \"from src.Decoders.vec_2_{modality.lower()} import {class_name}; dec = {class_name}()\"")
    print("  4. Train the decoder adapter using backtranslation")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new decoder module from boilerplate template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create audio decoder with Bark TTS
  python tools/create_decoder.py --modality audio --model suno/bark-small

  # Create video decoder (experimental)
  python tools/create_decoder.py --modality video --model microsoft/xclip-base-patch32

  # Create decoder marked as production-ready
  python tools/create_decoder.py --modality text --model facebook/bart-base --no-experimental
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
        help='HuggingFace model identifier (e.g., suno/bark-small)'
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
        '--no-experimental',
        action='store_true',
        help='Do not mark decoder as experimental'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Custom output path (default: src/Decoders/vec_2_{modality}.py)'
    )

    args = parser.parse_args()

    # Validate modality name
    if not args.modality.isalnum():
        print(f"Error: Modality name must be alphanumeric, got: {args.modality}")
        sys.exit(1)

    # Generate decoder
    output_path = generate_decoder(
        modality=args.modality,
        base_model=args.model,
        output_dim=args.output_dim,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        experimental=not args.no_experimental,
        output_path=args.output
    )

    print()
    print("✓ Decoder generation complete!")


if __name__ == "__main__":
    main()
