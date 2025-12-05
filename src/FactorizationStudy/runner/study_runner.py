"""
Study Runner for orchestrating full factorization studies.

Handles:
- Running multiple experiments sequentially
- Resume capability for interrupted studies
- Aggregating results across experiments
- Generating final reports
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.FactorizationStudy.config.study_config import StudyConfig, ExperimentConfig
from src.FactorizationStudy.config.config_generator import ConfigGenerator
from src.FactorizationStudy.runner.experiment import ExperimentRunner, ExperimentResults


class StudyRunner:
    """
    Orchestrates a complete factorization study with multiple experiments.
    """

    def __init__(
        self,
        study_config: StudyConfig,
        data_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize study runner.

        Args:
            study_config: Study configuration
            data_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        self.study_config = study_config
        self.data_loader = data_loader
        self.val_loader = val_loader

        # Generate experiments if not already done
        if not study_config.experiments:
            generator = ConfigGenerator(study_config)
            generator.generate()
            print(f"\n[StudyRunner] Generated {len(study_config.experiments)} experiments")

        # Setup directories
        self.study_dir = study_config.get_study_dir()
        self.study_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.completed_experiments: List[str] = []
        self.failed_experiments: List[str] = []
        self.results: Dict[str, ExperimentResults] = {}

        # Load any existing progress
        self._load_progress()

    def _load_progress(self) -> None:
        """Load progress from a previous run."""
        progress_file = self.study_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, "r") as f:
                progress = json.load(f)
                self.completed_experiments = progress.get("completed", [])
                self.failed_experiments = progress.get("failed", [])
            print(f"[StudyRunner] Loaded progress: {len(self.completed_experiments)} completed, "
                  f"{len(self.failed_experiments)} failed")

    def _save_progress(self) -> None:
        """Save current progress."""
        progress_file = self.study_dir / "progress.json"
        progress = {
            "completed": self.completed_experiments,
            "failed": self.failed_experiments,
            "last_updated": datetime.now().isoformat(),
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    def get_pending_experiments(self) -> List[ExperimentConfig]:
        """Get list of experiments that haven't been run yet."""
        completed_set = set(self.completed_experiments + self.failed_experiments)
        return [
            exp for exp in self.study_config.experiments
            if exp.experiment_id not in completed_set
        ]

    def run_experiment(self, exp_config: ExperimentConfig) -> ExperimentResults:
        """
        Run a single experiment.

        Args:
            exp_config: Experiment configuration

        Returns:
            ExperimentResults
        """
        runner = ExperimentRunner(
            experiment_config=exp_config,
            study_config=self.study_config,
            data_loader=self.data_loader,
            val_loader=self.val_loader,
        )

        try:
            results = runner.run()
            self.completed_experiments.append(exp_config.experiment_id)
            self.results[exp_config.experiment_id] = results
            return results
        except Exception as e:
            print(f"[StudyRunner] Experiment {exp_config.experiment_id} failed: {e}")
            self.failed_experiments.append(exp_config.experiment_id)
            raise

    def run(self, max_experiments: Optional[int] = None) -> Dict[str, ExperimentResults]:
        """
        Run all pending experiments.

        Args:
            max_experiments: Maximum number of experiments to run (None = all)

        Returns:
            Dictionary of experiment results
        """
        pending = self.get_pending_experiments()

        if max_experiments:
            pending = pending[:max_experiments]

        total = len(pending)
        if total == 0:
            print("[StudyRunner] No pending experiments to run")
            return self.results

        print(f"\n{'='*70}")
        print(f"FACTORIZATION STUDY: {self.study_config.study_name}")
        print(f"{'='*70}")
        print(f"Total experiments: {len(self.study_config.experiments)}")
        print(f"Already completed: {len(self.completed_experiments)}")
        print(f"Previously failed: {len(self.failed_experiments)}")
        print(f"Pending: {total}")
        print(f"{'='*70}\n")

        # Save study config
        self.study_config.save()

        start_time = time.time()

        for i, exp_config in enumerate(pending, 1):
            print(f"\n[{i}/{total}] Running experiment: {exp_config.experiment_id}")

            try:
                self.run_experiment(exp_config)
            except Exception as e:
                print(f"[StudyRunner] Continuing after failure...")

            # Save progress after each experiment
            self._save_progress()

        # Final summary
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

        return self.results

    def _print_summary(self, elapsed_seconds: float) -> None:
        """Print study summary."""
        hours = elapsed_seconds / 3600

        print(f"\n{'='*70}")
        print("STUDY SUMMARY")
        print(f"{'='*70}")
        print(f"Total time: {hours:.1f} hours")
        print(f"Completed: {len(self.completed_experiments)}")
        print(f"Failed: {len(self.failed_experiments)}")

        if self.results:
            # Find best result
            best_exp = min(self.results.values(), key=lambda r: r.best_loss)
            print(f"\nBest experiment: {best_exp.experiment_id}")
            print(f"  Best loss: {best_exp.best_loss:.4f}")
            print(f"  Config: {best_exp.config.get('semantic_dim')}D, "
                  f"h{best_exp.config.get('encoder_adapter', {}).get('hidden_size')}_"
                  f"l{best_exp.config.get('encoder_adapter', {}).get('hidden_layers')}")

        print(f"{'='*70}\n")

    def get_results_dataframe(self) -> List[Dict[str, Any]]:
        """
        Get results as a list of dictionaries (for analysis).

        Returns:
            List of result dictionaries
        """
        rows = []
        for exp_id, result in self.results.items():
            row = {
                "experiment_id": exp_id,
                "semantic_dim": result.config.get("semantic_dim"),
                "hidden_size": result.config.get("encoder_adapter", {}).get("hidden_size"),
                "hidden_layers": result.config.get("encoder_adapter", {}).get("hidden_layers"),
                "init_strategy": result.config.get("init_strategy"),
                "final_loss": result.final_loss,
                "best_loss": result.best_loss,
                "best_epoch": result.best_epoch,
                "completed": result.completed,
            }

            if result.resource_summary:
                row["peak_gpu_mb"] = result.resource_summary.peak_gpu_memory_mb
                row["avg_epoch_time"] = result.resource_summary.avg_epoch_time_seconds
                row["total_time"] = result.resource_summary.total_training_time_seconds

            rows.append(row)

        return rows

    def generate_reports(self) -> None:
        """Generate summary reports for the study."""
        from src.FactorizationStudy.reporting.report_generator import ReportGenerator

        generator = ReportGenerator(
            study_config=self.study_config,
            results=self.results,
        )
        generator.generate_all_reports()


def run_quick_study(
    data_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    study_name: str = "quick_test",
) -> Dict[str, ExperimentResults]:
    """
    Convenience function to run a quick test study.

    Args:
        data_loader: Training data loader
        val_loader: Validation data loader
        study_name: Name for the study

    Returns:
        Dictionary of results
    """
    from src.FactorizationStudy.config.config_generator import generate_quick_configs

    config = generate_quick_configs(study_name)
    runner = StudyRunner(config, data_loader, val_loader)
    return runner.run()


def run_full_sweep(
    data_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    study_name: str = "full_sweep",
) -> Dict[str, ExperimentResults]:
    """
    Convenience function to run a full sweep study.

    Args:
        data_loader: Training data loader
        val_loader: Validation data loader
        study_name: Name for the study

    Returns:
        Dictionary of results
    """
    from src.FactorizationStudy.config.config_generator import generate_full_sweep_configs

    config = generate_full_sweep_configs(study_name)
    runner = StudyRunner(config, data_loader, val_loader)
    return runner.run()
