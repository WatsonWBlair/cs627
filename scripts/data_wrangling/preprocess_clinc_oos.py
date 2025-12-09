"""
CLINC-OOS Dataset Wrangling for Intent Classification

Loads and preprocesses the CLINC-OOS intent classification dataset.
Outputs triplets for contrastive learning (same intent = positive).

Usage:
    python scripts/data_wrangling/preprocess_clinc_oos.py
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Iterator
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class CLINCWrangler(StreamingWranglerBase):
    """
    Process CLINC-OOS intent classification dataset.

    Generates triplets where:
    - Anchor and positive have the same intent
    - Negative has a different intent
    """

    def __init__(
        self,
        output_dir: str = "data/clinc/",
        checkpoint_interval: int = 5000,
        samples_per_intent: int = 50
    ):
        """
        Initialize CLINC wrangler.

        Args:
            output_dir: Directory to save outputs
            checkpoint_interval: Samples between checkpoints
            samples_per_intent: Max triplets per intent class
        """
        super().__init__("clinc", output_dir, checkpoint_interval)
        self.samples_per_intent = samples_per_intent
        self.negative_pool = NegativePool(max_size=5000)

    def process(self) -> List[Dict[str, str]]:
        """
        Process CLINC-OOS dataset.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print("CLINC-OOS Data Extraction (Streaming)")
        print("=" * 80)

        # Load all splits with streaming
        print("Loading dataset...")
        train = load_dataset("clinc_oos", "plus", split="train", streaming=True)

        # Group by intent
        intent_groups = defaultdict(list)

        print("Grouping by intent...")
        for sample in tqdm(train, desc="Processing"):
            text = sample['text']
            intent = sample['intent']

            self.negative_pool.add(text)
            intent_groups[intent].append(text)

        # Generate triplets
        print(f"\nFound {len(intent_groups)} intent classes")
        print("Generating triplets...")

        intents = list(intent_groups.keys())
        for intent in tqdm(intents, desc="Generating"):
            texts = intent_groups[intent]
            if len(texts) < 2:
                continue

            # Generate triplets for this intent
            count = 0
            for i, anchor in enumerate(texts):
                if count >= self.samples_per_intent:
                    break

                # Select positive from same intent
                positive_candidates = [t for t in texts if t != anchor]
                if not positive_candidates:
                    continue
                positive = random.choice(positive_candidates)

                # Select negative from different intent
                other_intent = random.choice([i for i in intents if i != intent])
                negative = random.choice(intent_groups[other_intent])

                self.triplets.append({
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative
                })
                count += 1

        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total triplets: {len(self.triplets)}")
        print(f"Intent classes: {len(intent_groups)}")

        return self.triplets


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CLINC-OOS for contrastive learning"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/clinc/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--samples-per-intent',
        type=int,
        default=50,
        help='Max triplets per intent class'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()
    random.seed(args.seed)

    wrangler = CLINCWrangler(
        output_dir=args.output_dir,
        samples_per_intent=args.samples_per_intent
    )

    output_path = wrangler.run(resume=True)

    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
