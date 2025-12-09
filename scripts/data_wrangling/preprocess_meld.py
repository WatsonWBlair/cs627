"""
MELD Dataset Wrangling for Emotion Classification

Loads and preprocesses the MELD (Multimodal EmotionLines Dataset) for
contrastive learning. Outputs triplets where same emotion = positive pair.

Usage:
    python scripts/data_wrangling/preprocess_meld.py
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Iterator
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class MELDWrangler(StreamingWranglerBase):
    """
    Process MELD emotion classification dataset.

    Generates triplets where:
    - Anchor and positive have the same emotion
    - Negative has a different emotion
    """

    # MELD emotions: anger, disgust, fear, joy, neutral, sadness, surprise
    EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    def __init__(
        self,
        output_dir: str = "data/meld/",
        checkpoint_interval: int = 5000,
        samples_per_emotion: int = 100
    ):
        """
        Initialize MELD wrangler.

        Args:
            output_dir: Directory to save outputs
            checkpoint_interval: Samples between checkpoints
            samples_per_emotion: Max triplets per emotion class
        """
        super().__init__("meld", output_dir, checkpoint_interval)
        self.samples_per_emotion = samples_per_emotion
        self.negative_pool = NegativePool(max_size=5000)

    def process(self) -> List[Dict[str, str]]:
        """
        Process MELD dataset.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print("MELD Data Extraction (Streaming)")
        print("=" * 80)

        # Load all splits with streaming
        print("Loading dataset...")
        train = load_dataset("declare-lab/MELD", split="train", streaming=True)

        # Group by emotion
        emotion_groups = defaultdict(list)

        print("Grouping by emotion...")
        for sample in tqdm(train, desc="Processing"):
            text = sample['Utterance']
            emotion = sample['Emotion']

            # Skip empty utterances
            if not text or not text.strip():
                continue

            self.negative_pool.add(text)
            emotion_groups[emotion].append(text)

        # Generate triplets
        print(f"\nFound {len(emotion_groups)} emotion classes:")
        for emotion, texts in emotion_groups.items():
            print(f"  {emotion}: {len(texts)} samples")

        print("\nGenerating triplets...")
        emotions = list(emotion_groups.keys())

        for emotion in tqdm(emotions, desc="Generating"):
            texts = emotion_groups[emotion]
            if len(texts) < 2:
                continue

            # Generate triplets for this emotion
            count = 0
            for i, anchor in enumerate(texts):
                if count >= self.samples_per_emotion:
                    break

                # Select positive from same emotion
                positive_candidates = [t for t in texts if t != anchor]
                if not positive_candidates:
                    continue
                positive = random.choice(positive_candidates)

                # Select negative from different emotion
                other_emotion = random.choice([e for e in emotions if e != emotion])
                if not emotion_groups[other_emotion]:
                    negative = self.negative_pool.sample(exclude={anchor, positive})
                else:
                    negative = random.choice(emotion_groups[other_emotion])

                if negative:
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
        print(f"Emotion classes: {len(emotion_groups)}")

        return self.triplets


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MELD for contrastive learning"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/meld/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--samples-per-emotion',
        type=int,
        default=100,
        help='Max triplets per emotion class'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()
    random.seed(args.seed)

    wrangler = MELDWrangler(
        output_dir=args.output_dir,
        samples_per_emotion=args.samples_per_emotion
    )

    output_path = wrangler.run(resume=True)

    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
