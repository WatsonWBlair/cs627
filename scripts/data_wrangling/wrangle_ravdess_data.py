"""
RAVDESS Dataset Wrangler for Emotional Speech Analysis

Loads RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
via HuggingFace datasets for contrastive learning.

Creates triplets where:
- Same emotion (different actors) = positive pair
- Different emotion = negative pair

RAVDESS contains:
- 1,440 speech clips from 24 professional actors (12M, 12F)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- 2 intensity levels: normal, strong
- Filename encodes all metadata

Usage:
    python scripts/data_wrangling/wrangle_ravdess_data.py
    python scripts/data_wrangling/wrangle_ravdess_data.py --include-song
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class RAVDESSWrangler(StreamingWranglerBase):
    """
    Process RAVDESS emotional speech dataset via HuggingFace.

    Generates triplets where:
    - Anchor and positive have the same emotion (different actors)
    - Negative has a different emotion

    Note: RAVDESS has no text transcripts. Triplets use emotion-actor-intensity
    labels as text identifiers for contrastive learning on audio embeddings.
    """

    HF_DATASET = "narad/ravdess"

    # RAVDESS emotion codes (from filename)
    EMOTIONS = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }

    # Intensity codes
    INTENSITIES = {
        1: 'normal',
        2: 'strong'
    }

    # Modality codes
    MODALITIES = {
        1: 'full_av',      # Audio-Visual
        2: 'video_only',   # Video-only
        3: 'audio_only'    # Audio-only
    }

    # Channel codes
    CHANNELS = {
        1: 'speech',
        2: 'song'
    }

    def __init__(
        self,
        output_dir: str = "data/ravdess/",
        checkpoint_interval: int = 1000,
        samples_per_emotion: int = 100,
        include_song: bool = False
    ):
        """
        Initialize RAVDESS wrangler.

        Args:
            output_dir: Directory to save outputs
            checkpoint_interval: Samples between checkpoints
            samples_per_emotion: Max triplets per emotion class
            include_song: Include song clips (default: speech only)
        """
        super().__init__("ravdess", output_dir, checkpoint_interval)
        self.samples_per_emotion = samples_per_emotion
        self.include_song = include_song
        self.negative_pool = NegativePool(max_size=2000)

    def _parse_filename(self, filename: str) -> Optional[Dict[str, int]]:
        """
        Parse RAVDESS filename to extract metadata.

        Filename format: MM-CC-EE-II-SS-RR-AA.wav
        - MM: Modality (01=full-AV, 02=video-only, 03=audio-only)
        - CC: Vocal channel (01=speech, 02=song)
        - EE: Emotion (01-08)
        - II: Emotional intensity (01=normal, 02=strong)
        - SS: Statement (01="Kids are talking...", 02="Dogs are sitting...")
        - RR: Repetition (01=1st, 02=2nd)
        - AA: Actor (01-24, odd=male, even=female)

        Args:
            filename: RAVDESS filename (e.g., "03-01-06-01-02-01-12.wav")

        Returns:
            Dict with parsed metadata or None if invalid
        """
        try:
            # Remove path and extension
            base = filename.split('/')[-1].split('\\')[-1]
            base = base.replace('.wav', '').replace('.mp4', '')

            parts = base.split('-')
            if len(parts) != 7:
                return None

            return {
                'modality': int(parts[0]),
                'channel': int(parts[1]),
                'emotion': int(parts[2]),
                'intensity': int(parts[3]),
                'statement': int(parts[4]),
                'repetition': int(parts[5]),
                'actor': int(parts[6])
            }
        except (ValueError, IndexError):
            return None

    def _create_label(self, metadata: Dict[str, int]) -> str:
        """
        Create a text label from parsed metadata.

        Since RAVDESS has no transcripts, we create semantic labels
        combining emotion, actor, and intensity for triplet generation.

        Args:
            metadata: Parsed filename metadata

        Returns:
            Label string (e.g., "angry_actor12_strong")
        """
        emotion = self.EMOTIONS.get(metadata['emotion'], 'unknown')
        intensity = self.INTENSITIES.get(metadata['intensity'], 'normal')
        actor = metadata['actor']
        gender = 'M' if actor % 2 == 1 else 'F'

        return f"{emotion}_actor{actor:02d}{gender}_{intensity}"

    def process(self) -> List[Dict[str, str]]:
        """
        Process RAVDESS dataset.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print("RAVDESS Data Extraction")
        print("=" * 80)

        print(f"Loading {self.HF_DATASET}...")
        print(f"Include song: {self.include_song}")
        print(f"Samples per emotion: {self.samples_per_emotion}")

        try:
            # Load dataset
            dataset = load_dataset(self.HF_DATASET, split="train")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        # Group by emotion
        emotion_groups: Dict[int, List[Tuple[str, Dict]]] = defaultdict(list)
        sample_count = 0
        skipped = 0

        print("\nGrouping samples by emotion...")
        for sample in tqdm(dataset, desc="Processing"):
            sample_count += 1

            # Get filename from sample
            # Dataset may use 'path', 'file', 'audio' field
            filename = None
            if 'path' in sample:
                filename = sample['path']
            elif 'file' in sample:
                filename = sample['file']
            elif 'audio' in sample and isinstance(sample['audio'], dict):
                filename = sample['audio'].get('path', '')

            if not filename:
                skipped += 1
                continue

            # Parse filename
            metadata = self._parse_filename(filename)
            if not metadata:
                skipped += 1
                continue

            # Filter by channel if needed
            if not self.include_song and metadata['channel'] == 2:
                skipped += 1
                continue

            # Create label
            label = self._create_label(metadata)

            # Add to negative pool
            self.negative_pool.add(label)

            # Group by emotion
            emotion_groups[metadata['emotion']].append((label, metadata))

        # Report statistics
        print(f"\nProcessed {sample_count} samples ({skipped} skipped)")
        print(f"Found {len(emotion_groups)} emotion classes:")
        for emotion_code in sorted(emotion_groups.keys()):
            emotion_name = self.EMOTIONS.get(emotion_code, 'unknown')
            count = len(emotion_groups[emotion_code])
            print(f"  {emotion_name} ({emotion_code}): {count} samples")

        # Generate triplets
        print("\nGenerating triplets...")
        emotions_with_data = [e for e in emotion_groups if len(emotion_groups[e]) >= 2]

        if not emotions_with_data:
            print("Warning: No emotions have enough samples for triplet generation")
            return []

        for emotion_code in tqdm(emotions_with_data, desc="Generating"):
            samples = emotion_groups[emotion_code]
            emotion_name = self.EMOTIONS.get(emotion_code, 'unknown')

            # Generate triplets for this emotion
            count = 0
            used_pairs = set()

            for label, metadata in samples:
                if count >= self.samples_per_emotion:
                    break

                # Select positive: same emotion, different actor
                anchor_actor = metadata['actor']
                positive_candidates = [
                    (l, m) for l, m in samples
                    if m['actor'] != anchor_actor
                ]

                if not positive_candidates:
                    # Fall back to any different sample
                    positive_candidates = [(l, m) for l, m in samples if l != label]

                if not positive_candidates:
                    continue

                pos_label, pos_meta = random.choice(positive_candidates)

                # Avoid duplicate pairs
                pair_key = tuple(sorted([label, pos_label]))
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)

                # Select negative: different emotion
                other_emotions = [e for e in emotions_with_data if e != emotion_code]
                if other_emotions:
                    other_emotion = random.choice(other_emotions)
                    neg_samples = emotion_groups[other_emotion]
                    neg_label, _ = random.choice(neg_samples)
                else:
                    neg_label = self.negative_pool.sample(exclude={label, pos_label})

                if neg_label:
                    self.triplets.append({
                        'anchor': label,
                        'positive': pos_label,
                        'negative': neg_label
                    })
                    count += 1

        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total triplets: {len(self.triplets)}")
        print(f"Samples processed: {sample_count}")
        print(f"Emotion classes: {len(emotions_with_data)}")

        # Show example triplet
        if self.triplets:
            print("\nExample triplet:")
            ex = self.triplets[0]
            print(f"  Anchor:   {ex['anchor']}")
            print(f"  Positive: {ex['positive']}")
            print(f"  Negative: {ex['negative']}")

        return self.triplets


def main():
    parser = argparse.ArgumentParser(
        description="Extract training triplets from RAVDESS dataset"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/ravdess/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--samples-per-emotion',
        type=int,
        default=100,
        help='Max triplets per emotion class'
    )
    parser.add_argument(
        '--include-song',
        action='store_true',
        help='Include song clips (default: speech only)'
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

    wrangler = RAVDESSWrangler(
        output_dir=args.output_dir,
        samples_per_emotion=args.samples_per_emotion,
        include_song=args.include_song
    )

    output_path = wrangler.run(resume=not args.no_resume)

    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
