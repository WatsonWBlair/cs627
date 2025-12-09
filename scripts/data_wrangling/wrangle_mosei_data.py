"""
CMU-MOSEI Dataset Wrangler for Sentiment and Emotion Analysis

Loads CMU-MOSEI via HuggingFace datasets for contrastive learning.
Creates triplets based on emotion labels (same emotion = positive pair).

CMU-MOSEI is 10x larger than MOSI with:
- 22,852 segments across 1,000 YouTube videos
- Sentiment labels [-3, 3]
- 6 emotion labels: happy, sad, anger, surprise, disgust, fear

Usage:
    python scripts/data_wrangling/wrangle_mosei_data.py
    python scripts/data_wrangling/wrangle_mosei_data.py --samples-per-emotion 1000
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class MOSEIWrangler(StreamingWranglerBase):
    """
    Process CMU-MOSEI dataset via HuggingFace.

    Generates triplets where:
    - Anchor and positive have the same dominant emotion
    - Negative has a different dominant emotion

    Uses emotion scores to determine dominant emotion per utterance.
    """

    HF_DATASET = "reeha-parkar/cmu-mosei-comp-seq"

    # MOSEI emotion labels (scores 0-3)
    EMOTIONS = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']

    # Column mappings for the dataset
    EMOTION_COLUMNS = {
        'happy': 'happy',
        'sad': 'sad',
        'anger': 'anger',
        'surprise': 'surprise',
        'disgust': 'disgust',
        'fear': 'fear'
    }

    def __init__(
        self,
        output_dir: str = "data/mosei/",
        checkpoint_interval: int = 5000,
        samples_per_emotion: int = 500,
        emotion_threshold: float = 0.5,
        sentiment_threshold: float = 1.0
    ):
        """
        Initialize MOSEI wrangler.

        Args:
            output_dir: Directory to save outputs
            checkpoint_interval: Samples between checkpoints
            samples_per_emotion: Max triplets per emotion class
            emotion_threshold: Minimum score to consider emotion present
            sentiment_threshold: Minimum sentiment magnitude for grouping
        """
        super().__init__("mosei", output_dir, checkpoint_interval)
        self.samples_per_emotion = samples_per_emotion
        self.emotion_threshold = emotion_threshold
        self.sentiment_threshold = sentiment_threshold
        self.negative_pool = NegativePool(max_size=10000)

    def _get_dominant_emotion(self, sample: Dict) -> Optional[str]:
        """
        Determine dominant emotion from sample's emotion scores.

        Args:
            sample: Dataset sample with emotion scores

        Returns:
            Name of dominant emotion or None if no clear dominant
        """
        emotion_scores = {}
        for emotion in self.EMOTIONS:
            col = self.EMOTION_COLUMNS.get(emotion, emotion)
            score = sample.get(col, 0)
            if score is not None:
                emotion_scores[emotion] = float(score)

        if not emotion_scores:
            return None

        # Find max emotion above threshold
        max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        if max_emotion[1] >= self.emotion_threshold:
            return max_emotion[0]

        return None

    def _extract_text(self, sample: Dict) -> Optional[str]:
        """
        Extract text from sample.

        Args:
            sample: Dataset sample

        Returns:
            Text content or None if empty
        """
        # Try different possible text fields
        text = sample.get('text', '')
        if not text:
            text = sample.get('transcript', '')
        if not text:
            text = sample.get('sentence', '')

        if text and isinstance(text, str) and text.strip():
            return text.strip()
        return None

    def process(self) -> List[Dict[str, str]]:
        """
        Process MOSEI dataset.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print("CMU-MOSEI Data Extraction (Streaming)")
        print("=" * 80)

        print(f"Loading {self.HF_DATASET}...")
        print(f"Emotion threshold: {self.emotion_threshold}")
        print(f"Samples per emotion: {self.samples_per_emotion}")

        try:
            # Load dataset - try different configurations
            try:
                dataset = load_dataset(
                    self.HF_DATASET,
                    split="train",
                    streaming=True
                )
            except Exception:
                # Try without streaming if streaming fails
                print("Streaming not available, loading full dataset...")
                dataset = load_dataset(self.HF_DATASET, split="train")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nTrying alternative dataset...")

            # Fallback to alternative MOSEI dataset
            try:
                dataset = load_dataset(
                    "reeha-parkar/custom-unaligned-CMU-MOSEI",
                    split="train",
                    streaming=True
                )
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return []

        # Group by dominant emotion
        emotion_groups: Dict[str, List[str]] = defaultdict(list)
        sample_count = 0
        skipped = 0

        print("\nGrouping samples by emotion...")
        for sample in tqdm(dataset, desc="Processing"):
            sample_count += 1

            # Extract text
            text = self._extract_text(sample)
            if not text:
                skipped += 1
                continue

            # Add to negative pool
            self.negative_pool.add(text)

            # Determine dominant emotion
            dominant = self._get_dominant_emotion(sample)
            if dominant:
                emotion_groups[dominant].append(text)

            # Periodic progress
            if sample_count % 5000 == 0:
                print(f"  Processed {sample_count} samples...")

        # Report statistics
        print(f"\nProcessed {sample_count} samples ({skipped} skipped)")
        print(f"Found {len(emotion_groups)} emotion classes:")
        for emotion in self.EMOTIONS:
            count = len(emotion_groups.get(emotion, []))
            print(f"  {emotion}: {count} samples")

        # Generate triplets
        print("\nGenerating triplets...")
        emotions_with_data = [e for e in self.EMOTIONS if len(emotion_groups[e]) >= 2]

        if not emotions_with_data:
            print("Warning: No emotions have enough samples for triplet generation")
            return []

        for emotion in tqdm(emotions_with_data, desc="Generating"):
            texts = emotion_groups[emotion]

            # Generate triplets for this emotion
            count = 0
            for anchor in texts:
                if count >= self.samples_per_emotion:
                    break

                # Select positive from same emotion
                positive_candidates = [t for t in texts if t != anchor]
                if not positive_candidates:
                    continue
                positive = random.choice(positive_candidates)

                # Select negative from different emotion
                other_emotions = [e for e in emotions_with_data if e != emotion]
                if other_emotions:
                    other_emotion = random.choice(other_emotions)
                    negative_candidates = emotion_groups[other_emotion]
                    if negative_candidates:
                        negative = random.choice(negative_candidates)
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
                    count += 1

        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total triplets: {len(self.triplets)}")
        print(f"Samples processed: {sample_count}")
        print(f"Emotion classes with data: {len(emotions_with_data)}")

        return self.triplets


def main():
    parser = argparse.ArgumentParser(
        description="Extract training triplets from CMU-MOSEI dataset"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/mosei/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--samples-per-emotion',
        type=int,
        default=500,
        help='Max triplets per emotion class'
    )
    parser.add_argument(
        '--emotion-threshold',
        type=float,
        default=0.5,
        help='Minimum emotion score to consider emotion present'
    )
    parser.add_argument(
        '--sentiment-threshold',
        type=float,
        default=1.0,
        help='Minimum sentiment magnitude for grouping'
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

    wrangler = MOSEIWrangler(
        output_dir=args.output_dir,
        samples_per_emotion=args.samples_per_emotion,
        emotion_threshold=args.emotion_threshold,
        sentiment_threshold=args.sentiment_threshold
    )

    output_path = wrangler.run(resume=not args.no_resume)

    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
