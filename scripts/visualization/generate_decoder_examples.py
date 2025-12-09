"""
Generate Decoder Output Examples

Demonstrates decoder capabilities by generating outputs from sample inputs.
Creates text, audio, and image examples from semantic vectors.

Usage:
    python scripts/generate_decoder_examples.py

    # With specific examples:
    python scripts/generate_decoder_examples.py --text "Hello world"
    python scripts/generate_decoder_examples.py --audio data/sample.wav
    python scripts/generate_decoder_examples.py --list-examples
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.Encoders import Text_to_Vec

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory
OUTPUT_DIR = "results/decoder_examples/"


def generate_text_examples(texts=None):
    """
    Generate text decoder examples.

    Encodes input text to semantic vector, then decodes back to text.
    Shows the encode-decode cycle and semantic fidelity.
    """
    from src.Decoders import Vec_to_Text

    print("\n" + "=" * 60)
    print("TEXT DECODER EXAMPLES")
    print("=" * 60)

    # Default example texts
    if texts is None:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "I really enjoyed the movie last night.",
            "The weather today is absolutely beautiful.",
            "This new technology could change everything.",
            "I'm feeling happy and excited about the future."
        ]

    # Load models
    print("\nLoading models...")
    encoder = Text_to_Vec().to(DEVICE)
    encoder.eval()

    decoder = Vec_to_Text().to(DEVICE)
    decoder.eval()

    print(f"  [OK] Text encoder loaded")
    print(f"  [OK] Text decoder loaded")

    # Generate examples
    print("\n" + "-" * 60)
    print("EXAMPLES")
    print("-" * 60)

    results = []
    for i, text in enumerate(texts):
        print(f"\nExample {i+1}:")
        print(f"  Input:  \"{text}\"")

        # Encode
        with torch.no_grad():
            semantic_vec = encoder(text, device=DEVICE)

            # Decode
            reconstructed = decoder(semantic_vec, device=DEVICE)

            # Compute semantic similarity
            reconstructed_vec = encoder(reconstructed, device=DEVICE)
            similarity = torch.nn.functional.cosine_similarity(
                semantic_vec, reconstructed_vec, dim=1
            ).item()

        print(f"  Output: \"{reconstructed}\"")
        print(f"  Semantic Similarity: {similarity:.4f}")

        results.append({
            'input': text,
            'output': reconstructed,
            'semantic_similarity': similarity
        })

    # Summary
    avg_similarity = np.mean([r['semantic_similarity'] for r in results])
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"  Average Semantic Similarity: {avg_similarity:.4f}")

    return results


def generate_audio_examples(texts=None, output_dir=None):
    """
    Generate audio decoder examples.

    Encodes input text to semantic vector, then generates audio.
    Saves audio files to output directory.
    """
    try:
        from src.Decoders import Vec_to_Audio
    except ImportError as e:
        print(f"\n[WARNING] Audio decoder not available: {e}")
        print("  Train the audio decoder first: python src/Training/train_audio_decoder.py")
        return []

    import soundfile as sf

    print("\n" + "=" * 60)
    print("AUDIO DECODER EXAMPLES")
    print("=" * 60)

    # Default example texts
    if texts is None:
        texts = [
            "Hello, how are you today?",
            "The weather is nice.",
            "I'm excited about this."
        ]

    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "audio")
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    print("\nLoading models...")
    encoder = Text_to_Vec().to(DEVICE)
    encoder.eval()

    try:
        decoder = Vec_to_Audio().to(DEVICE)
        decoder.eval()
        print(f"  [OK] Audio decoder loaded")
    except Exception as e:
        print(f"  [WARNING] Audio decoder failed to load: {e}")
        print("  Train the audio decoder first")
        return []

    # Generate examples
    print("\n" + "-" * 60)
    print("GENERATING AUDIO")
    print("-" * 60)

    results = []
    for i, text in enumerate(texts):
        print(f"\nExample {i+1}: \"{text}\"")

        try:
            # Encode
            with torch.no_grad():
                semantic_vec = encoder(text, device=DEVICE)

                # Decode to audio
                audio = decoder(semantic_vec, device=DEVICE)

            # Save audio
            filename = f"audio_example_{i+1}.wav"
            filepath = os.path.join(output_dir, filename)
            sf.write(filepath, audio, decoder.sample_rate)

            print(f"  [OK] Saved: {filepath}")
            print(f"  Duration: {len(audio) / decoder.sample_rate:.2f}s")

            results.append({
                'input': text,
                'output_file': filepath,
                'duration': len(audio) / decoder.sample_rate
            })

        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")

    return results


def generate_image_examples(texts=None, output_dir=None):
    """
    Generate image decoder examples.

    Encodes input text to semantic vector, then generates image.
    Saves image files to output directory.
    """
    try:
        from src.Decoders import Vec_to_Image
    except ImportError as e:
        print(f"\n[WARNING] Image decoder not available: {e}")
        return []

    from PIL import Image

    print("\n" + "=" * 60)
    print("IMAGE DECODER EXAMPLES")
    print("=" * 60)

    # Default example texts
    if texts is None:
        texts = [
            "A beautiful sunset over the ocean",
            "A cat sitting on a windowsill",
            "A mountain landscape with snow"
        ]

    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    print("\nLoading models...")
    encoder = Text_to_Vec().to(DEVICE)
    encoder.eval()

    try:
        decoder = Vec_to_Image().to(DEVICE)
        decoder.eval()
        print(f"  [OK] Image decoder loaded")
    except Exception as e:
        print(f"  [WARNING] Image decoder failed to load: {e}")
        print("  This decoder is experimental")
        return []

    # Generate examples
    print("\n" + "-" * 60)
    print("GENERATING IMAGES")
    print("-" * 60)

    results = []
    for i, text in enumerate(texts):
        print(f"\nExample {i+1}: \"{text}\"")

        try:
            # Encode
            with torch.no_grad():
                semantic_vec = encoder(text, device=DEVICE)

                # Decode to image
                image = decoder(semantic_vec, device=DEVICE)

            # Save image
            filename = f"image_example_{i+1}.png"
            filepath = os.path.join(output_dir, filename)

            if isinstance(image, Image.Image):
                image.save(filepath)
            else:
                # Convert numpy array to PIL Image
                Image.fromarray(image).save(filepath)

            print(f"  [OK] Saved: {filepath}")

            results.append({
                'input': text,
                'output_file': filepath
            })

        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")

    return results


def main():
    """Main function to generate all decoder examples."""
    parser = argparse.ArgumentParser(description='Generate decoder output examples')
    parser.add_argument('--text', type=str, nargs='*',
                       help='Custom text inputs for examples')
    parser.add_argument('--modalities', type=str, nargs='*',
                       default=['text', 'audio', 'image'],
                       choices=['text', 'audio', 'image'],
                       help='Which modalities to generate examples for')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for generated examples')
    parser.add_argument('--list-examples', action='store_true',
                       help='List default example inputs')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("DECODER OUTPUT EXAMPLE GENERATOR")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Output directory: {args.output_dir}")
    print(f"Modalities: {args.modalities}")
    print("=" * 60)

    # List examples if requested
    if args.list_examples:
        print("\nDefault Example Texts:")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "I really enjoyed the movie last night.",
            "The weather today is absolutely beautiful.",
            "This new technology could change everything.",
            "I'm feeling happy and excited about the future."
        ]
        for i, t in enumerate(texts, 1):
            print(f"  {i}. \"{t}\"")
        return 0

    # Custom texts
    texts = args.text if args.text else None

    # Generate examples for each modality
    all_results = {}

    if 'text' in args.modalities:
        all_results['text'] = generate_text_examples(texts)

    if 'audio' in args.modalities:
        audio_dir = os.path.join(args.output_dir, 'audio')
        all_results['audio'] = generate_audio_examples(texts, audio_dir)

    if 'image' in args.modalities:
        image_dir = os.path.join(args.output_dir, 'images')
        all_results['image'] = generate_image_examples(texts, image_dir)

    # Save results summary
    import json
    summary_path = os.path.join(args.output_dir, 'generation_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': DEVICE,
        'modalities': args.modalities,
        'results': {}
    }

    for modality, results in all_results.items():
        summary['results'][modality] = results

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
