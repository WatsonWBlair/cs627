"""
Streaming Utilities for HuggingFace Dataset Wranglers

Provides base classes and shared utilities for all data wrangling scripts.
All wranglers inherit from StreamingWranglerBase for consistent:
- Checkpoint/resume support
- Unified output format (triplets)
- Streaming processing

Usage:
    from streaming_utils import StreamingWranglerBase, parse_segment_id

    class MyWrangler(StreamingWranglerBase):
        def process(self):
            ...
"""

import json
import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
from collections import deque


class StreamingWranglerBase(ABC):
    """
    Base class for HuggingFace streaming dataset wrappers.

    Provides:
    - Checkpoint/resume functionality
    - Unified triplet output format
    - Progress tracking
    """

    def __init__(
        self,
        name: str,
        output_dir: str,
        checkpoint_interval: int = 5000
    ):
        """
        Initialize base wrangler.

        Args:
            name: Dataset name (used for checkpoint files)
            output_dir: Directory to save outputs
            checkpoint_interval: Samples between checkpoints
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval

        # Processing state
        self._processed_count = 0
        self._last_index = -1
        self._task_progress: Dict[str, str] = {}

        # Output storage
        self.triplets: List[Dict[str, str]] = []
        self.all_sentences: set = set()  # For negative sampling

    @property
    def checkpoint_path(self) -> Path:
        """Path to checkpoint file."""
        return self.output_dir / f"{self.name}_checkpoint.json"

    @property
    def output_path(self) -> Path:
        """Path to output triplets file."""
        return self.output_dir / f"{self.name}_triplets.json"

    def state_dict(self) -> Dict[str, Any]:
        """
        Return current processing state for checkpointing.

        Returns:
            Dictionary containing processing state
        """
        return {
            'name': self.name,
            'processed_count': self._processed_count,
            'last_index': self._last_index,
            'task_progress': self._task_progress,
            'triplets_count': len(self.triplets),
            'sentences_count': len(self.all_sentences),
            'timestamp': datetime.now().isoformat()
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Resume from saved state.

        Args:
            state: State dictionary from previous run
        """
        self._processed_count = state.get('processed_count', 0)
        self._last_index = state.get('last_index', -1)
        self._task_progress = state.get('task_progress', {})

    def save_checkpoint(self) -> None:
        """Save current state to checkpoint file."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.state_dict(), f, indent=2)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint if exists.

        Returns:
            State dictionary or None if no checkpoint
        """
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return None

    def clear_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def save_triplets(self, triplets: Optional[List[Dict[str, str]]] = None) -> Path:
        """
        Save triplets in unified format.

        Args:
            triplets: List of triplet dicts, or use self.triplets if None

        Returns:
            Path to saved file
        """
        if triplets is None:
            triplets = self.triplets

        output = {
            'triplets': triplets,
            'metadata': {
                'source': self.name,
                'total_samples': len(triplets),
                'created_at': datetime.now().isoformat()
            }
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Clear checkpoint on successful save
        self.clear_checkpoint()

        return self.output_path

    def add_triplet(
        self,
        anchor: str,
        positive: str,
        negative: Optional[str] = None
    ) -> None:
        """
        Add a triplet to the collection.

        Args:
            anchor: Anchor text
            positive: Positive (similar) text
            negative: Negative (dissimilar) text, or sample from pool
        """
        # Track sentences for negative sampling
        self.all_sentences.add(anchor)
        self.all_sentences.add(positive)

        # Sample negative if not provided
        if negative is None:
            negative = self._sample_negative(anchor, positive)

        if negative:
            self.triplets.append({
                'anchor': anchor,
                'positive': positive,
                'negative': negative
            })

    def _sample_negative(self, anchor: str, positive: str) -> Optional[str]:
        """
        Sample a random negative from the sentence pool.

        Args:
            anchor: Anchor text (to avoid)
            positive: Positive text (to avoid)

        Returns:
            Random negative sentence or None if pool too small
        """
        candidates = self.all_sentences - {anchor, positive}
        if len(candidates) < 1:
            return None
        return random.choice(list(candidates))

    @abstractmethod
    def process(self) -> List[Dict[str, str]]:
        """
        Process the dataset and return triplets.

        Must be implemented by subclasses.

        Returns:
            List of triplet dictionaries
        """
        pass

    def run(self, resume: bool = True) -> Path:
        """
        Run the wrangling pipeline with optional resume.

        Args:
            resume: Whether to resume from checkpoint

        Returns:
            Path to output file
        """
        # Try to resume from checkpoint
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                print(f"Resuming from checkpoint: {checkpoint['processed_count']} samples")
                self.load_state_dict(checkpoint)

        # Process dataset
        triplets = self.process()

        # Save results
        output_path = self.save_triplets(triplets)
        print(f"Saved {len(triplets)} triplets to: {output_path}")

        return output_path


def parse_segment_id(seg_id: str) -> Tuple[str, int]:
    """
    Parse MOSI segment ID into video ID and segment number.

    Args:
        seg_id: Segment ID (e.g., 'BvYR0L6f2Ig[2]')

    Returns:
        Tuple of (video_id, segment_number)

    Example:
        >>> parse_segment_id('BvYR0L6f2Ig[2]')
        ('BvYR0L6f2Ig', 2)
    """
    if '[' in seg_id:
        parts = seg_id.split('[')
        video_id = parts[0]
        segment_num = int(parts[1].rstrip(']'))
        return video_id, segment_num
    return seg_id, 0


def segment_id_to_filename(seg_id: str, extension: str = '') -> str:
    """
    Convert segment ID to filesystem-safe filename.

    Args:
        seg_id: Segment ID (e.g., 'BvYR0L6f2Ig[2]')
        extension: File extension (e.g., '.wav')

    Returns:
        Filename (e.g., 'BvYR0L6f2Ig_2.wav')

    Example:
        >>> segment_id_to_filename('BvYR0L6f2Ig[2]', '.wav')
        'BvYR0L6f2Ig_2.wav'
    """
    video_id, segment_num = parse_segment_id(seg_id)
    base = f"{video_id}_{segment_num}" if segment_num > 0 else video_id
    return f"{base}{extension}"


def split_data(
    data: List[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split data into train/val/test sets.

    Args:
        data: List of samples to split
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val, test) lists
    """
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        shuffled[:train_end],
        shuffled[train_end:val_end],
        shuffled[val_end:]
    )


class NegativePool:
    """
    Efficient pool for negative sampling with bounded memory.

    Maintains a sliding window of recent samples for negative mining.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize negative pool.

        Args:
            max_size: Maximum number of samples to keep
        """
        self.max_size = max_size
        self._pool: deque = deque(maxlen=max_size)

    def add(self, text: str) -> None:
        """Add a text to the pool."""
        self._pool.append(text)

    def sample(self, exclude: Optional[set] = None) -> Optional[str]:
        """
        Sample a random text from the pool.

        Args:
            exclude: Set of texts to exclude from sampling

        Returns:
            Random text or None if pool empty/exhausted
        """
        if not self._pool:
            return None

        exclude = exclude or set()
        candidates = [t for t in self._pool if t not in exclude]

        if not candidates:
            return None

        return random.choice(candidates)

    def __len__(self) -> int:
        return len(self._pool)
