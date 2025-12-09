"""
LAION Dataset Wrangler for Large-Scale Image-Text Training

Loads LAION via HuggingFace datasets for contrastive learning.
Creates triplets from caption pairs with similar CLIP scores.

Usage:
    python scripts/data_wrangling/wrangle_laion_data.py
    python scripts/data_wrangling/wrangle_laion_data.py --subset 2b-en
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class LAIONWrangler(StreamingWranglerBase):
    """
    Process LAION dataset via HuggingFace.

    Generates triplets from CLIP-filtered image-text pairs.
    Uses high CLIP scores as positive indicators.
    """

    HF_DATASETS = {
        '400m': 'laion/laion400m',
        '2b-en': 'laion/laion2b-en',
        '5b': 'laion/laion5b'
    }

    def __init__(
        self,
        output_dir: str = "data/laion/",
        subset: str = "400m",
        checkpoint_interval: int = 5000,
        max_samples: int = 50000,
        min_score: float = 0.25,
        min_width: int = 256,
        min_height: int = 256
    ):
        """
        Initialize LAION wrangler.

        Args:
            output_dir: Directory to save outputs
            subset: Which subset ('400m', '2b-en', '5b')
            checkpoint_interval: Samples between checkpoints
            max_samples: Maximum samples to process
            min_score: Minimum CLIP similarity score
            min_width: Minimum image width
            min_height: Minimum image height
        """
        super().__init__(f"laion_{subset}", output_dir, checkpoint_interval)
        self.subset = subset
        self.max_samples = max_samples
        self.min_score = min_score
        self.min_width = min_width
        self.min_height = min_height
        self.negative_pool = NegativePool(max_size=10000)

    def _extract_keywords(self, caption: str) -> set:
        """Extract key concepts from caption for grouping."""
        words = caption.lower().split()
        return {w for w in words if len(w) > 4 and w.isalpha()}

    def process(self) -> List[Dict[str, str]]:
        """
        Process LAION dataset.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print(f"LAION Data Extraction ({self.subset.upper()})")
        print("=" * 80)

        hf_dataset = self.HF_DATASETS.get(self.subset)
        if not hf_dataset:
            print(f"Error: Unknown subset {self.subset}")
            return []

        print(f"Loading {hf_dataset} with streaming...")
        print(f"Filters: min_score={self.min_score}, min_size={self.min_width}x{self.min_height}")

        try:
            dataset = load_dataset(
                hf_dataset,
                split="train",
                streaming=True
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        # Collect captions and group by similarity
        all_captions = []
        keyword_groups: Dict[str, List[str]] = defaultdict(list)
        count = 0

        print("Collecting captions...")
        for sample in tqdm(dataset, desc="Processing"):
            if count >= self.max_samples * 2:
                break

            # Apply quality filters
            similarity = sample.get('similarity', 0.0)
            if similarity < self.min_score:
                continue

            width = sample.get('width', 0)
            height = sample.get('height', 0)
            if width < self.min_width or height < self.min_height:
                continue

            # Filter NSFW
            nsfw = sample.get('nsfw', 'UNLIKELY')
            if nsfw not in ['UNLIKELY', 'VERY_UNLIKELY']:
                continue

            caption = sample.get('caption', sample.get('text', ''))
            if not caption or len(caption.split()) < 3:
                continue

            all_captions.append(caption)
            self.negative_pool.add(caption)

            # Group by keywords for positive pairing
            keywords = self._extract_keywords(caption)
            for keyword in keywords:
                if len(keyword_groups[keyword]) < 100:
                    keyword_groups[keyword].append(caption)

            count += 1

        # Generate triplets
        print(f"\nCollected {len(all_captions)} high-quality captions")
        print(f"Found {len(keyword_groups)} keyword groups")
        print("Generating triplets...")

        triplet_count = 0
        used_pairs = set()

        sorted_groups = sorted(
            keyword_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for keyword, captions in tqdm(sorted_groups, desc="Generating"):
            if len(captions) < 2:
                continue

            for anchor in captions:
                if triplet_count >= self.max_samples:
                    break

                # Select positive (same keyword)
                positive_candidates = [c for c in captions if c != anchor]
                if not positive_candidates:
                    continue

                positive = random.choice(positive_candidates)

                # Skip duplicate pairs
                pair_key = tuple(sorted([anchor, positive]))
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)

                # Select negative
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
        description="Extract training triplets from LAION dataset"
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='400m',
        choices=['400m', '2b-en', '5b'],
        help='LAION subset to use'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/laion/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50000,
        help='Maximum triplets to generate'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.25,
        help='Minimum CLIP similarity score'
    )
    parser.add_argument(
        '--min-width',
        type=int,
        default=256,
        help='Minimum image width'
    )
    parser.add_argument(
        '--min-height',
        type=int,
        default=256,
        help='Minimum image height'
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

    wrangler = LAIONWrangler(
        output_dir=args.output_dir,
        subset=args.subset,
        max_samples=args.max_samples,
        min_score=args.min_score,
        min_width=args.min_width,
        min_height=args.min_height
    )

    output_path = wrangler.run(resume=not args.no_resume)

    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
