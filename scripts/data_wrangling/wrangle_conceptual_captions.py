"""
Conceptual Captions Dataset Wrangler for Image-Text Training

Loads Conceptual Captions via HuggingFace datasets for contrastive learning.
Creates triplets where different captions describe similar concepts.

Usage:
    python scripts/data_wrangling/wrangle_conceptual_captions.py
    python scripts/data_wrangling/wrangle_conceptual_captions.py --dataset cc12m
"""

import argparse
import random
import re
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class ConceptualCaptionsWrangler(StreamingWranglerBase):
    """
    Process Conceptual Captions dataset via HuggingFace.

    Generates triplets from image-text pairs by grouping
    similar captions together.
    """

    HF_DATASETS = {
        'cc3m': 'conceptual_captions',
        'cc12m': 'google-research-datasets/conceptual_12m'
    }

    def __init__(
        self,
        output_dir: str = "data/conceptual_captions/",
        dataset: str = "cc3m",
        checkpoint_interval: int = 5000,
        max_samples: int = 50000,
        min_caption_length: int = 3,
        max_caption_length: int = 256
    ):
        """
        Initialize Conceptual Captions wrangler.

        Args:
            output_dir: Directory to save outputs
            dataset: Which dataset ('cc3m' or 'cc12m')
            checkpoint_interval: Samples between checkpoints
            max_samples: Maximum samples to process
            min_caption_length: Minimum caption length in words
            max_caption_length: Maximum caption length in words
        """
        super().__init__(f"cc_{dataset}", output_dir, checkpoint_interval)
        self.dataset = dataset
        self.max_samples = max_samples
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length
        self.negative_pool = NegativePool(max_size=10000)

    def _clean_caption(self, caption: str) -> str:
        """Clean and normalize caption text."""
        if not caption:
            return ""

        # Remove extra whitespace
        caption = ' '.join(caption.split())

        # Fix common HTML entities
        caption = caption.replace('&amp;', '&')
        caption = caption.replace('&lt;', '<')
        caption = caption.replace('&gt;', '>')
        caption = caption.replace('&quot;', '"')
        caption = caption.replace('&#39;', "'")

        # Remove URLs from captions
        caption = re.sub(r'http\S+|www\.\S+', '', caption)

        return caption.strip()

    def _filter_caption(self, caption: str) -> bool:
        """Check if caption meets quality requirements."""
        if not caption:
            return False
        words = caption.split()
        return self.min_caption_length <= len(words) <= self.max_caption_length

    def _extract_keywords(self, caption: str) -> set:
        """Extract key nouns/concepts from caption for grouping."""
        # Simple approach: use words longer than 4 chars
        words = caption.lower().split()
        return {w for w in words if len(w) > 4 and w.isalpha()}

    def process(self) -> List[Dict[str, str]]:
        """
        Process Conceptual Captions dataset.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print(f"Conceptual Captions Data Extraction ({self.dataset.upper()})")
        print("=" * 80)

        hf_dataset = self.HF_DATASETS.get(self.dataset)
        if not hf_dataset:
            print(f"Error: Unknown dataset {self.dataset}")
            return []

        print(f"Loading {hf_dataset} with streaming...")
        try:
            dataset = load_dataset(
                hf_dataset,
                split="train",
                streaming=True
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        # Collect captions and group by keywords
        all_captions = []
        keyword_groups: Dict[str, List[str]] = defaultdict(list)
        count = 0

        print("Collecting captions...")
        for sample in tqdm(dataset, desc="Processing"):
            if count >= self.max_samples * 2:
                break

            caption = sample.get('caption', '')
            caption = self._clean_caption(caption)

            if not self._filter_caption(caption):
                continue

            all_captions.append(caption)
            self.negative_pool.add(caption)

            # Group by keywords for positive pairing
            keywords = self._extract_keywords(caption)
            for keyword in keywords:
                if len(keyword_groups[keyword]) < 100:  # Limit group size
                    keyword_groups[keyword].append(caption)

            count += 1

        # Generate triplets from keyword groups
        print(f"\nCollected {len(all_captions)} captions")
        print(f"Found {len(keyword_groups)} keyword groups")
        print("Generating triplets...")

        triplet_count = 0
        used_pairs = set()

        # Sort groups by size for better coverage
        sorted_groups = sorted(
            keyword_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for keyword, captions in tqdm(sorted_groups, desc="Generating"):
            if len(captions) < 2:
                continue

            for i, anchor in enumerate(captions):
                if triplet_count >= self.max_samples:
                    break

                # Select positive (same keyword group)
                positive_candidates = [c for c in captions if c != anchor]
                if not positive_candidates:
                    continue

                positive = random.choice(positive_candidates)

                # Skip if we've seen this pair
                pair_key = tuple(sorted([anchor, positive]))
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)

                # Select negative (different keywords)
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
        print(f"Captions processed: {len(all_captions)}")

        return self.triplets


def main():
    parser = argparse.ArgumentParser(
        description="Extract training triplets from Conceptual Captions"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cc3m',
        choices=['cc3m', 'cc12m'],
        help='Which dataset to use'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/conceptual_captions/',
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
        default=3,
        help='Minimum caption length in words'
    )
    parser.add_argument(
        '--max-caption-length',
        type=int,
        default=256,
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

    wrangler = ConceptualCaptionsWrangler(
        output_dir=args.output_dir,
        dataset=args.dataset,
        max_samples=args.max_samples,
        min_caption_length=args.min_caption_length,
        max_caption_length=args.max_caption_length
    )

    output_path = wrangler.run(resume=not args.no_resume)

    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
