"""
MS COCO Dataset Wrangler for Image-Text Training

Loads MS COCO via HuggingFace datasets for contrastive learning.
Creates triplets where different captions for the same image are positives.

Usage:
    python scripts/data_wrangling/wrangle_coco_data.py
    python scripts/data_wrangling/wrangle_coco_data.py --max-samples 10000
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Iterator
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class COCOWrangler(StreamingWranglerBase):
    """
    Process MS COCO dataset via HuggingFace.

    Generates triplets where:
    - Anchor and positive are different captions for the same image
    - Negative is a caption from a different image
    """

    HF_DATASET = "HuggingFaceM4/COCO"

    def __init__(
        self,
        output_dir: str = "data/coco/",
        checkpoint_interval: int = 5000,
        max_samples: int = 50000,
        min_caption_length: int = 5,
        max_caption_length: int = 100
    ):
        """
        Initialize COCO wrangler.

        Args:
            output_dir: Directory to save outputs
            checkpoint_interval: Samples between checkpoints
            max_samples: Maximum samples to process
            min_caption_length: Minimum caption length in words
            max_caption_length: Maximum caption length in words
        """
        super().__init__("coco", output_dir, checkpoint_interval)
        self.max_samples = max_samples
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length
        self.negative_pool = NegativePool(max_size=10000)

    def _filter_caption(self, caption: str) -> bool:
        """Check if caption meets length requirements."""
        if not caption:
            return False
        words = caption.split()
        return self.min_caption_length <= len(words) <= self.max_caption_length

    def process(self) -> List[Dict[str, str]]:
        """
        Process COCO dataset.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print("COCO Data Extraction (HuggingFace)")
        print("=" * 80)

        print("Loading dataset with streaming...")
        try:
            dataset = load_dataset(
                self.HF_DATASET,
                split="train",
                streaming=True
            )
        except Exception as e:
            print(f"Error loading COCO from HuggingFace: {e}")
            print("Trying alternative dataset ID...")
            dataset = load_dataset(
                "detection-datasets/coco",
                split="train",
                streaming=True
            )

        # Group captions by image
        image_captions: Dict[str, List[str]] = defaultdict(list)
        count = 0

        print("Collecting captions...")
        for sample in tqdm(dataset, desc="Processing"):
            if count >= self.max_samples * 2:  # Collect extra for filtering
                break

            # Handle different COCO dataset schemas
            image_id = str(sample.get('image_id', sample.get('id', count)))

            # Get captions - different datasets have different formats
            captions = sample.get('captions', [])
            if not captions and 'caption' in sample:
                captions = [sample['caption']]
            if not captions and 'sentences' in sample:
                captions = [s.get('raw', s.get('text', '')) for s in sample['sentences']]

            for caption in captions:
                if isinstance(caption, dict):
                    caption = caption.get('raw', caption.get('text', ''))

                if self._filter_caption(caption):
                    image_captions[image_id].append(caption)
                    self.negative_pool.add(caption)

            count += 1

        # Generate triplets from images with multiple captions
        print(f"\nFound {len(image_captions)} images with valid captions")
        print("Generating triplets...")

        triplet_count = 0
        image_ids = list(image_captions.keys())

        for image_id in tqdm(image_ids, desc="Generating"):
            captions = image_captions[image_id]
            if len(captions) < 2:
                continue

            # Create triplets from caption pairs
            for i, anchor in enumerate(captions):
                if triplet_count >= self.max_samples:
                    break

                # Select positive (different caption, same image)
                positive_candidates = [c for j, c in enumerate(captions) if j != i]
                if not positive_candidates:
                    continue
                positive = random.choice(positive_candidates)

                # Select negative (caption from different image)
                other_images = [img for img in image_ids if img != image_id]
                if other_images:
                    other_image = random.choice(other_images)
                    if image_captions[other_image]:
                        negative = random.choice(image_captions[other_image])
                    else:
                        negative = self.negative_pool.sample(exclude={anchor, positive})
                else:
                    negative = self.negative_pool.sample(exclude={anchor, positive})

                if negative:
                    self.triplets.append({
                        'anchor': anchor,
                        'positive': positive,
                        'negative': negative
                    })
                    triplet_count += 1

            if triplet_count >= self.max_samples:
                break

        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total triplets: {len(self.triplets)}")
        print(f"Images processed: {len(image_captions)}")

        return self.triplets


def main():
    parser = argparse.ArgumentParser(
        description="Extract training triplets from MS COCO dataset"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/coco/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50000,
        help='Maximum triplets to generate'
    )
    parser.add_argument(
        '--min-caption-length',
        type=int,
        default=5,
        help='Minimum caption length in words'
    )
    parser.add_argument(
        '--max-caption-length',
        type=int,
        default=100,
        help='Maximum caption length in words'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )

    args = parser.parse_args()
    random.seed(args.seed)

    wrangler = COCOWrangler(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        min_caption_length=args.min_caption_length,
        max_caption_length=args.max_caption_length
    )

    output_path = wrangler.run(resume=not args.no_resume)

    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
