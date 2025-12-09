"""
LibriSpeech Dataset Wrangler for Audio-Text Training

Loads LibriSpeech via HuggingFace datasets for contrastive learning.
Creates text triplets from transcripts (same speaker = positive).

Usage:
    python scripts/data_wrangling/wrangle_librispeech_data.py
    python scripts/data_wrangling/wrangle_librispeech_data.py --subset clean
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Iterator
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class LibriSpeechWrangler(StreamingWranglerBase):
    """
    Process LibriSpeech dataset via HuggingFace.

    Generates triplets where:
    - Anchor and positive are transcripts from the same speaker
    - Negative is a transcript from a different speaker
    """

    HF_DATASET = "librispeech_asr"

    # Available configs
    CONFIGS = {
        'clean': ['train.clean.100', 'train.clean.360'],
        'other': ['train.other.500'],
        'all': ['train.clean.100', 'train.clean.360', 'train.other.500'],
        'small': ['train.clean.100']
    }

    def __init__(
        self,
        output_dir: str = "data/librispeech/",
        subset: str = "clean",
        checkpoint_interval: int = 5000,
        max_samples: int = 50000,
        samples_per_speaker: int = 100
    ):
        """
        Initialize LibriSpeech wrangler.

        Args:
            output_dir: Directory to save outputs
            subset: Dataset subset ('clean', 'other', 'all', 'small')
            checkpoint_interval: Samples between checkpoints
            max_samples: Maximum total samples to process
            samples_per_speaker: Max triplets per speaker
        """
        super().__init__("librispeech", output_dir, checkpoint_interval)
        self.subset = subset
        self.max_samples = max_samples
        self.samples_per_speaker = samples_per_speaker
        self.negative_pool = NegativePool(max_size=10000)

    def process(self) -> List[Dict[str, str]]:
        """
        Process LibriSpeech dataset.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print("LibriSpeech Data Extraction (HuggingFace)")
        print("=" * 80)
        print(f"Subset: {self.subset}")

        # Get configs to process
        configs = self.CONFIGS.get(self.subset, [self.subset])
        print(f"Processing: {', '.join(configs)}")

        # Group transcripts by speaker
        speaker_texts: Dict[str, List[str]] = defaultdict(list)
        total_count = 0

        for config in configs:
            if self._task_progress.get(config) == 'complete':
                print(f"\n[{config}] Already complete, skipping...")
                continue

            self._task_progress[config] = 'in_progress'
            print(f"\n[{config}] Loading with streaming...")

            try:
                # LibriSpeech on HF uses split names like "train.clean.100"
                dataset = load_dataset(
                    self.HF_DATASET,
                    config.replace('.', '-') if '.' in config else config,
                    split="train" if "train" in config else config,
                    streaming=True,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"[{config}] Error loading: {e}")
                # Try alternative format
                try:
                    dataset = load_dataset(
                        self.HF_DATASET,
                        split=config,
                        streaming=True,
                        trust_remote_code=True
                    )
                except Exception as e2:
                    print(f"[{config}] Alternative also failed: {e2}")
                    continue

            count = 0
            for sample in tqdm(dataset, desc=f"[{config}]"):
                if total_count >= self.max_samples * 2:
                    break

                text = sample.get('text', '')
                if not text:
                    continue

                # Get speaker ID
                speaker_id = str(sample.get('speaker_id', sample.get('id', '').split('-')[0]))

                speaker_texts[speaker_id].append(text)
                self.negative_pool.add(text)

                count += 1
                total_count += 1

            self._task_progress[config] = 'complete'
            self.save_checkpoint()
            print(f"[{config}] Processed {count} samples")

            if total_count >= self.max_samples * 2:
                break

        # Generate triplets
        print(f"\nFound {len(speaker_texts)} speakers")
        print("Generating triplets...")

        speakers = list(speaker_texts.keys())
        triplet_count = 0

        for speaker in tqdm(speakers, desc="Generating"):
            texts = speaker_texts[speaker]
            if len(texts) < 2:
                continue

            count = 0
            for i, anchor in enumerate(texts):
                if count >= self.samples_per_speaker:
                    break
                if triplet_count >= self.max_samples:
                    break

                # Select positive (same speaker)
                positive_candidates = [t for t in texts if t != anchor]
                if not positive_candidates:
                    continue
                positive = random.choice(positive_candidates)

                # Select negative (different speaker)
                other_speakers = [s for s in speakers if s != speaker]
                if other_speakers:
                    other_speaker = random.choice(other_speakers)
                    if speaker_texts[other_speaker]:
                        negative = random.choice(speaker_texts[other_speaker])
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
                    triplet_count += 1

            if triplet_count >= self.max_samples:
                break

        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total triplets: {len(self.triplets)}")
        print(f"Speakers: {len(speaker_texts)}")

        return self.triplets


def main():
    parser = argparse.ArgumentParser(
        description="Extract training triplets from LibriSpeech dataset"
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='small',
        choices=['clean', 'other', 'all', 'small'],
        help='Which subset to process'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/librispeech/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50000,
        help='Maximum triplets to generate'
    )
    parser.add_argument(
        '--samples-per-speaker',
        type=int,
        default=100,
        help='Max triplets per speaker'
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

    wrangler = LibriSpeechWrangler(
        output_dir=args.output_dir,
        subset=args.subset,
        max_samples=args.max_samples,
        samples_per_speaker=args.samples_per_speaker
    )

    output_path = wrangler.run(resume=not args.no_resume)

    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
