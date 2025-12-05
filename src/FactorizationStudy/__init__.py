"""
Factorization Study Framework for Semantic Vector Space Architecture

This module provides an automated hyperparameter sweep framework to study
performance impacts of:
- Semantic vector dimensions (256, 512, 1024, 2048)
- Encoder adapter configurations (hidden size, layers)
- Decoder adapter configurations (hidden size, layers)
- Encoder weight initialization strategies

Usage:
    python scripts/run_factorization_study.py --preset quick
    python scripts/run_factorization_study.py --preset full_sweep
    python scripts/run_factorization_study.py --resume
"""

from .config.study_config import (
    AdapterConfig,
    ExperimentConfig,
    StudyConfig,
    StudyPreset,
)
from .config.config_generator import ConfigGenerator
from .runner.experiment import ExperimentRunner
from .runner.study_runner import StudyRunner
from .evaluation.metrics_collector import MetricsCollector
from .reporting.report_generator import ReportGenerator

__all__ = [
    "AdapterConfig",
    "ExperimentConfig",
    "StudyConfig",
    "StudyPreset",
    "ConfigGenerator",
    "ExperimentRunner",
    "StudyRunner",
    "MetricsCollector",
    "ReportGenerator",
]
