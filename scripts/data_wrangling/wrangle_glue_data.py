"""
GLUE Dataset Wrangling for Encoder Training

Extracts sentence pairs and NLI data from GLUE benchmark tasks for contrastive learning.
Uses HuggingFace datasets with streaming for memory efficiency.

Supported tasks:
- MRPC: Paraphrase detection (sentence pairs)
- QQP: Quora question pairs (duplicate detection)
- STS-B: Semantic textual similarity (sentence pairs with scores)
- MNLI: Natural language inference (premise-hypothesis pairs)
- QNLI: Question-answering NLI (question-answer pairs)

Usage:
    python scripts/data_wrangling/wrangle_glue_data.py
    python scripts/data_wrangling/wrangle_glue_data.py --tasks mrpc qqp stsb
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Iterator
from tqdm import tqdm

from datasets import load_dataset

from streaming_utils import StreamingWranglerBase, NegativePool


class GLUEWrangler(StreamingWranglerBase):
    """
    Extract training triplets from GLUE benchmark tasks.

    Uses HuggingFace datasets streaming for memory-efficient processing.
    Outputs triplets in unified format for contrastive learning.
    """

    TASKS = ['mrpc', 'qqp', 'stsb', 'mnli', 'qnli']

    def __init__(
        self,
        output_dir: str = "data/glue/",
        tasks: List[str] = None,
        checkpoint_interval: int = 5000,
        max_samples_per_task: int = 50000
    ):
        """
        Initialize GLUE wrangler.

        Args:
            output_dir: Directory to save outputs
            tasks: List of GLUE tasks to process (default: all)
            checkpoint_interval: Samples between checkpoints
            max_samples_per_task: Maximum samples per task (for large datasets)
        """
        super().__init__("glue", output_dir, checkpoint_interval)
        self.tasks = tasks or self.TASKS
        self.max_samples = max_samples_per_task
        self.negative_pool = NegativePool(max_size=10000)

    def process(self) -> List[Dict[str, str]]:
        """
        Process all configured GLUE tasks.

        Returns:
            List of triplet dictionaries
        """
        print("\n" + "=" * 80)
        print("GLUE Data Extraction (Streaming)")
        print("=" * 80)
        print(f"Tasks: {', '.join(self.tasks)}")
        print()

        stats = defaultdict(int)

        for task in self.tasks:
            # Skip if already completed (resume support)
            if self._task_progress.get(task) == 'complete':
                print(f"[{task}] Already complete, skipping...")
                continue

            self._task_progress[task] = 'in_progress'
            task_triplets = list(self._process_task(task))
            self.triplets.extend(task_triplets)
            stats[task] = len(task_triplets)

            self._task_progress[task] = 'complete'
            self.save_checkpoint()

            print(f"  [{task}] Generated {len(task_triplets)} triplets")

        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total triplets: {len(self.triplets)}")
        for task, count in stats.items():
            print(f"  {task}: {count}")

        return self.triplets

    def _process_task(self, task: str) -> Iterator[Dict[str, str]]:
        """
        Process a single GLUE task with streaming.

        Args:
            task: GLUE task name

        Yields:
            Triplet dictionaries
        """
        if task == 'mrpc':
            yield from self._process_mrpc()
        elif task == 'qqp':
            yield from self._process_qqp()
        elif task == 'stsb':
            yield from self._process_stsb()
        elif task == 'mnli':
            yield from self._process_mnli()
        elif task == 'qnli':
            yield from self._process_qnli()

    def _process_mrpc(self) -> Iterator[Dict[str, str]]:
        """Process MRPC paraphrase pairs."""
        print("  [mrpc] Loading with streaming...")
        dataset = load_dataset('glue', 'mrpc', split='train', streaming=True)

        positive_pairs = []
        negative_sentences = []

        for sample in tqdm(dataset, desc="  [mrpc] Processing"):
            sent1, sent2 = sample['sentence1'], sample['sentence2']
            label = sample['label']

            self.negative_pool.add(sent1)
            self.negative_pool.add(sent2)

            if label == 1:  # Paraphrase
                positive_pairs.append((sent1, sent2))
            else:
                negative_sentences.extend([sent1, sent2])

        # Generate triplets
        for anchor, positive in positive_pairs:
            negative = self._get_negative(anchor, positive, negative_sentences)
            if negative:
                yield {'anchor': anchor, 'positive': positive, 'negative': negative}

    def _process_qqp(self) -> Iterator[Dict[str, str]]:
        """Process QQP question pairs."""
        print("  [qqp] Loading with streaming...")
        dataset = load_dataset('glue', 'qqp', split='train', streaming=True)

        positive_pairs = []
        negative_sentences = []
        count = 0

        for sample in tqdm(dataset, desc="  [qqp] Processing"):
            if count >= self.max_samples:
                break

            q1, q2 = sample['question1'], sample['question2']
            if not q1 or not q2:
                continue

            label = sample['label']
            self.negative_pool.add(q1)
            self.negative_pool.add(q2)

            if label == 1:  # Duplicate
                positive_pairs.append((q1, q2))
            else:
                negative_sentences.extend([q1, q2])

            count += 1

        # Generate triplets
        for anchor, positive in positive_pairs:
            negative = self._get_negative(anchor, positive, negative_sentences)
            if negative:
                yield {'anchor': anchor, 'positive': positive, 'negative': negative}

    def _process_stsb(self) -> Iterator[Dict[str, str]]:
        """Process STS-B similarity pairs."""
        print("  [stsb] Loading with streaming...")
        dataset = load_dataset('glue', 'stsb', split='train', streaming=True)

        positive_pairs = []  # score >= 4.0
        negative_sentences = []  # score <= 2.0

        for sample in tqdm(dataset, desc="  [stsb] Processing"):
            sent1, sent2 = sample['sentence1'], sample['sentence2']
            score = sample['label']

            self.negative_pool.add(sent1)
            self.negative_pool.add(sent2)

            if score >= 4.0:
                positive_pairs.append((sent1, sent2))
            if score <= 2.0:
                negative_sentences.extend([sent1, sent2])

        # Generate triplets
        for anchor, positive in positive_pairs:
            negative = self._get_negative(anchor, positive, negative_sentences)
            if negative:
                yield {'anchor': anchor, 'positive': positive, 'negative': negative}

    def _process_mnli(self) -> Iterator[Dict[str, str]]:
        """Process MNLI NLI samples."""
        print("  [mnli] Loading with streaming...")
        dataset = load_dataset('glue', 'mnli', split='train', streaming=True)

        # Group by premise
        premise_groups = defaultdict(lambda: {'entail': [], 'contra': [], 'neutral': []})
        count = 0

        for sample in tqdm(dataset, desc="  [mnli] Processing"):
            if count >= self.max_samples:
                break

            premise = sample['premise']
            hypothesis = sample['hypothesis']
            label = sample['label']  # 0=entailment, 1=neutral, 2=contradiction

            self.negative_pool.add(premise)
            self.negative_pool.add(hypothesis)

            if label == 0:
                premise_groups[premise]['entail'].append(hypothesis)
            elif label == 2:
                premise_groups[premise]['contra'].append(hypothesis)
            else:
                premise_groups[premise]['neutral'].append(hypothesis)

            count += 1

        # Generate triplets: premise + entailment + contradiction
        for premise, groups in premise_groups.items():
            for entailment in groups['entail']:
                if groups['contra']:
                    negative = random.choice(groups['contra'])
                elif groups['neutral']:
                    negative = random.choice(groups['neutral'])
                else:
                    negative = self.negative_pool.sample(exclude={premise, entailment})

                if negative:
                    yield {'anchor': premise, 'positive': entailment, 'negative': negative}

    def _process_qnli(self) -> Iterator[Dict[str, str]]:
        """Process QNLI question-answer pairs."""
        print("  [qnli] Loading with streaming...")
        dataset = load_dataset('glue', 'qnli', split='train', streaming=True)

        positive_pairs = []
        negative_sentences = []

        for sample in tqdm(dataset, desc="  [qnli] Processing"):
            question = sample['question']
            sentence = sample['sentence']
            label = sample['label']  # 0=entailment (answers), 1=not

            self.negative_pool.add(question)
            self.negative_pool.add(sentence)

            if label == 0:  # Sentence answers the question
                positive_pairs.append((question, sentence))
            else:
                negative_sentences.append(sentence)

        # Sample subset to avoid overwhelming with QNLI data
        if len(positive_pairs) > self.max_samples // 2:
            positive_pairs = random.sample(positive_pairs, self.max_samples // 2)

        # Generate triplets
        for anchor, positive in positive_pairs:
            negative = self._get_negative(anchor, positive, negative_sentences)
            if negative:
                yield {'anchor': anchor, 'positive': positive, 'negative': negative}

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

        # Try task-specific candidates first
        if candidates:
            valid = [c for c in candidates if c not in exclude]
            if valid:
                return random.choice(valid)

        # Fall back to general pool
        return self.negative_pool.sample(exclude=exclude)


def main():
    parser = argparse.ArgumentParser(
        description="Extract training triplets from GLUE benchmark tasks"
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['mrpc', 'qqp', 'stsb', 'mnli', 'qnli'],
        choices=['mrpc', 'qqp', 'stsb', 'mnli', 'qnli'],
        help='GLUE tasks to process'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/glue/',
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
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint (default: True)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore checkpoint'
    )

    args = parser.parse_args()
    random.seed(args.seed)

    wrangler = GLUEWrangler(
        output_dir=args.output_dir,
        tasks=args.tasks,
        max_samples_per_task=args.max_samples
    )

    output_path = wrangler.run(resume=not args.no_resume)

    print(f"\nData saved to: {output_path}")
    print("Use with: python src/Training/train_adapters.py --data-dir", args.output_dir)


if __name__ == "__main__":
    main()
