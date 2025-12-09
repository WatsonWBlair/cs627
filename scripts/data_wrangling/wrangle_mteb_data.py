"""
MTEB Dataset Wrangling for Encoder Training

Extracts high-quality text pairs and triplets from MTEB benchmark datasets
for contrastive learning. Uses HuggingFace datasets with streaming.

Supported task categories:
- STS: Semantic Textual Similarity (sentence pairs with scores)
- Retrieval: Query-document pairs

Usage:
    python scripts/data_wrangling/wrangle_mteb_data.py
    python scripts/data_wrangling/wrangle_mteb_data.py --tasks sts
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Iterator
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class MTEBWrangler(StreamingWranglerBase):
    """
    Extract training triplets from MTEB benchmark tasks.

    Uses HuggingFace datasets streaming for memory-efficient processing.
    """

    # STS datasets with similarity scores
    STS_TASKS = [
        ("STSBenchmark", "sentence-transformers/stsb"),
        ("STS12", "mteb/sts12"),
        ("STS13", "mteb/sts13"),
        ("STS14", "mteb/sts14"),
        ("STS15", "mteb/sts15"),
        ("STS16", "mteb/sts16"),
    ]

    # Retrieval tasks with query-document pairs
    RETRIEVAL_TASKS = [
        ("MSMARCOHardNeg", "sentence-transformers/msmarco-hard-negatives"),
    ]

    def __init__(
        self,
        output_dir: str = "data/mteb/",
        tasks: List[str] = None,
        checkpoint_interval: int = 5000,
        max_samples_per_task: int = 50000
    ):
        """
        Initialize MTEB wrangler.

        Args:
            output_dir: Directory to save outputs
            tasks: Task categories to process ['sts', 'retrieval']
            checkpoint_interval: Samples between checkpoints
            max_samples_per_task: Maximum samples per task
        """
        super().__init__("mteb", output_dir, checkpoint_interval)
        self.task_categories = tasks or ['sts', 'retrieval']
        self.max_samples = max_samples_per_task
        self.negative_pool = NegativePool(max_size=10000)

    def process(self) -> List[Dict[str, str]]:
        """
        Process all configured MTEB tasks.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print("MTEB Data Extraction (Streaming)")
        print("=" * 80)
        print(f"Categories: {', '.join(self.task_categories)}")
        print()

        stats = defaultdict(int)

        # Process STS tasks
        if 'sts' in self.task_categories:
            print("[STS Tasks]")
            for task_name, dataset_id in self.STS_TASKS:
                if self._task_progress.get(task_name) == 'complete':
                    print(f"  [{task_name}] Already complete, skipping...")
                    continue

                self._task_progress[task_name] = 'in_progress'
                task_triplets = list(self._process_sts_task(task_name, dataset_id))
                self.triplets.extend(task_triplets)
                stats[task_name] = len(task_triplets)

                self._task_progress[task_name] = 'complete'
                self.save_checkpoint()

                print(f"  [{task_name}] Generated {len(task_triplets)} triplets")

        # Process retrieval tasks
        if 'retrieval' in self.task_categories:
            print("\n[Retrieval Tasks]")
            for task_name, dataset_id in self.RETRIEVAL_TASKS:
                if self._task_progress.get(task_name) == 'complete':
                    print(f"  [{task_name}] Already complete, skipping...")
                    continue

                self._task_progress[task_name] = 'in_progress'
                task_triplets = list(self._process_retrieval_task(task_name, dataset_id))
                self.triplets.extend(task_triplets)
                stats[task_name] = len(task_triplets)

                self._task_progress[task_name] = 'complete'
                self.save_checkpoint()

                print(f"  [{task_name}] Generated {len(task_triplets)} triplets")

        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total triplets: {len(self.triplets)}")
        for task, count in stats.items():
            print(f"  {task}: {count}")

        return self.triplets

    def _process_sts_task(
        self,
        task_name: str,
        dataset_id: str
    ) -> Iterator[Dict[str, str]]:
        """
        Process an STS task with streaming.

        Args:
            task_name: Task display name
            dataset_id: HuggingFace dataset ID

        Yields:
            Triplet dictionaries
        """
        print(f"  [{task_name}] Loading with streaming...")

        try:
            dataset = load_dataset(dataset_id, split='train', streaming=True)
        except Exception as e:
            print(f"  [{task_name}] Error loading: {e}")
            return

        positive_pairs = []  # score >= 0.7 (normalized)
        negative_sentences = []  # score < 0.3

        count = 0
        for sample in tqdm(dataset, desc=f"  [{task_name}]"):
            if count >= self.max_samples:
                break

            # Handle different column names
            sent1 = sample.get('sentence1', sample.get('text1', ''))
            sent2 = sample.get('sentence2', sample.get('text2', ''))
            score = sample.get('score', sample.get('label', 0.0))

            if not sent1 or not sent2:
                continue

            # Normalize score to [0, 1]
            if score > 1:
                score = score / 5.0

            self.negative_pool.add(sent1)
            self.negative_pool.add(sent2)

            if score >= 0.7:
                positive_pairs.append((sent1, sent2))
            if score < 0.3:
                negative_sentences.extend([sent1, sent2])

            count += 1

        # Generate triplets
        for anchor, positive in positive_pairs:
            negative = self._get_negative(anchor, positive, negative_sentences)
            if negative:
                yield {'anchor': anchor, 'positive': positive, 'negative': negative}

    def _process_retrieval_task(
        self,
        task_name: str,
        dataset_id: str
    ) -> Iterator[Dict[str, str]]:
        """
        Process a retrieval task with streaming.

        Args:
            task_name: Task display name
            dataset_id: HuggingFace dataset ID

        Yields:
            Triplet dictionaries
        """
        print(f"  [{task_name}] Loading with streaming...")

        try:
            dataset = load_dataset(dataset_id, split='train', streaming=True)
        except Exception as e:
            print(f"  [{task_name}] Error loading: {e}")
            return

        count = 0
        for sample in tqdm(dataset, desc=f"  [{task_name}]"):
            if count >= self.max_samples:
                break

            query = sample.get('query', '')
            positive = sample.get('positive', '')
            negatives = sample.get('negative', [])

            if not query or not positive:
                continue

            self.negative_pool.add(query)
            self.negative_pool.add(positive)

            # Get negative from the sample or pool
            if negatives:
                if isinstance(negatives, list) and negatives:
                    negative = negatives[0]
                else:
                    negative = negatives
            else:
                negative = self.negative_pool.sample(exclude={query, positive})

            if negative:
                yield {'anchor': query, 'positive': positive, 'negative': negative}

            count += 1

    def _get_negative(
        self,
        anchor: str,
        positive: str,
        candidates: List[str]
    ) -> str:
        """
        Get a negative sample avoiding anchor and positive.

        Args:
            anchor: Anchor text
            positive: Positive text
            candidates: List of candidate negatives

        Returns:
            Negative text or None
        """
        exclude = {anchor, positive}

        if candidates:
            valid = [c for c in candidates if c not in exclude]
            if valid:
                return random.choice(valid)

        return self.negative_pool.sample(exclude=exclude)


def main():
    parser = argparse.ArgumentParser(
        description="Extract training triplets from MTEB benchmark tasks"
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['sts', 'retrieval'],
        choices=['sts', 'retrieval'],
        help='Task categories to process'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/mteb/',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50000,
        help='Maximum samples per task'
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

    wrangler = MTEBWrangler(
        output_dir=args.output_dir,
        tasks=args.tasks,
        max_samples_per_task=args.max_samples
    )

    output_path = wrangler.run(resume=not args.no_resume)

    print(f"\nData saved to: {output_path}")
    print("Use with: python src/Training/train_adapters.py --data-dir", args.output_dir)


if __name__ == "__main__":
    main()
