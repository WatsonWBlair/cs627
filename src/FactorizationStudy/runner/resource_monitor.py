"""
Resource Monitor for tracking GPU memory and training time.

Collects system metrics during training for the factorization study.
"""

import time
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ResourceSnapshot:
    """A snapshot of resource usage at a point in time."""
    timestamp: float
    epoch: int
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    gpu_memory_max_allocated_mb: float = 0.0
    cpu_memory_mb: float = 0.0


@dataclass
class ResourceSummary:
    """Summary of resource usage across an experiment."""
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    total_training_time_seconds: float = 0.0
    avg_epoch_time_seconds: float = 0.0
    min_epoch_time_seconds: float = 0.0
    max_epoch_time_seconds: float = 0.0
    snapshots: List[ResourceSnapshot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "avg_gpu_memory_mb": self.avg_gpu_memory_mb,
            "total_training_time_seconds": self.total_training_time_seconds,
            "avg_epoch_time_seconds": self.avg_epoch_time_seconds,
            "min_epoch_time_seconds": self.min_epoch_time_seconds,
            "max_epoch_time_seconds": self.max_epoch_time_seconds,
            "num_snapshots": len(self.snapshots),
        }


class ResourceMonitor:
    """
    Monitors GPU memory and training time during experiments.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize resource monitor.

        Args:
            device: Device to monitor (cuda or cpu)
        """
        self.device = device
        self.has_cuda = torch.cuda.is_available() and device.startswith("cuda")

        self.snapshots: List[ResourceSnapshot] = []
        self.epoch_times: List[float] = []

        self.training_start_time: Optional[float] = None
        self.training_end_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None

        self.current_epoch = 0

    def start_training(self) -> None:
        """Mark the start of training."""
        self.training_start_time = time.time()
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()
        print(f"[ResourceMonitor] Training started at {datetime.now().strftime('%H:%M:%S')}")

    def end_training(self) -> None:
        """Mark the end of training."""
        self.training_end_time = time.time()
        if self.has_cuda:
            torch.cuda.synchronize()

    def start_epoch(self, epoch: int) -> None:
        """Mark the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def end_epoch(self) -> float:
        """
        Mark the end of an epoch and record metrics.

        Returns:
            Epoch duration in seconds
        """
        if self.has_cuda:
            torch.cuda.synchronize()

        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        # Take snapshot
        snapshot = self._take_snapshot()
        self.snapshots.append(snapshot)

        return epoch_time

    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            epoch=self.current_epoch,
        )

        if self.has_cuda:
            snapshot.gpu_memory_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            snapshot.gpu_memory_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            snapshot.gpu_memory_max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return snapshot

    def get_current_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if not self.has_cuda:
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 2)

    def get_peak_gpu_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        if not self.has_cuda:
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    def get_summary(self) -> ResourceSummary:
        """Get summary of resource usage."""
        summary = ResourceSummary(snapshots=self.snapshots)

        # GPU memory stats
        if self.snapshots:
            gpu_memories = [s.gpu_memory_allocated_mb for s in self.snapshots]
            summary.peak_gpu_memory_mb = max(gpu_memories) if gpu_memories else 0.0
            summary.avg_gpu_memory_mb = sum(gpu_memories) / len(gpu_memories) if gpu_memories else 0.0

        # Timing stats
        if self.training_start_time and self.training_end_time:
            summary.total_training_time_seconds = self.training_end_time - self.training_start_time

        if self.epoch_times:
            summary.avg_epoch_time_seconds = sum(self.epoch_times) / len(self.epoch_times)
            summary.min_epoch_time_seconds = min(self.epoch_times)
            summary.max_epoch_time_seconds = max(self.epoch_times)

        return summary

    def check_memory_limit(self, limit_mb: Optional[float]) -> bool:
        """
        Check if GPU memory usage exceeds limit.

        Args:
            limit_mb: Memory limit in MB (None means no limit)

        Returns:
            True if within limit, False if exceeded
        """
        if limit_mb is None or not self.has_cuda:
            return True

        current_mb = self.get_current_gpu_memory_mb()
        return current_mb <= limit_mb

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def print_status(self) -> None:
        """Print current resource status."""
        if self.has_cuda:
            current = self.get_current_gpu_memory_mb()
            peak = self.get_peak_gpu_memory_mb()
            print(f"[ResourceMonitor] GPU Memory: {current:.0f}MB (peak: {peak:.0f}MB)")

        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            print(f"[ResourceMonitor] Avg epoch time: {self.format_duration(avg_time)}")
