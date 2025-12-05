"""
Report Generator for Factorization Study results.

Generates:
- Summary report with best configurations
- Dimension analysis report
- Adapter configuration analysis report
- CSV exports for external analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.FactorizationStudy.config.study_config import StudyConfig
from src.FactorizationStudy.runner.experiment import ExperimentResults


class ReportGenerator:
    """
    Generates comprehensive reports from factorization study results.
    """

    def __init__(
        self,
        study_config: StudyConfig,
        results: Dict[str, ExperimentResults],
    ):
        """
        Initialize report generator.

        Args:
            study_config: Study configuration
            results: Dictionary mapping experiment IDs to results
        """
        self.study_config = study_config
        self.results = results
        self.output_dir = study_config.get_study_dir() / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_reports(self) -> None:
        """Generate all reports."""
        print(f"\n[ReportGenerator] Generating reports to {self.output_dir}")

        self.generate_summary_report()
        self.generate_dimension_analysis()
        self.generate_adapter_analysis()
        self.export_csv()

        print(f"[ReportGenerator] All reports generated")

    def generate_summary_report(self) -> str:
        """
        Generate summary report with key findings.

        Returns:
            Path to generated report
        """
        report = {
            "study_name": self.study_config.study_name,
            "generated_at": datetime.now().isoformat(),
            "total_experiments": len(self.results),
            "completed_experiments": sum(1 for r in self.results.values() if r.completed),
            "failed_experiments": sum(1 for r in self.results.values() if not r.completed),
        }

        # Find best overall configuration
        completed_results = [r for r in self.results.values() if r.completed]
        if completed_results:
            best = min(completed_results, key=lambda r: r.best_loss)
            report["best_overall"] = {
                "experiment_id": best.experiment_id,
                "best_loss": best.best_loss,
                "best_epoch": best.best_epoch,
                "config": {
                    "semantic_dim": best.config.get("semantic_dim"),
                    "hidden_size": best.config.get("encoder_adapter", {}).get("hidden_size"),
                    "hidden_layers": best.config.get("encoder_adapter", {}).get("hidden_layers"),
                    "init_strategy": best.config.get("init_strategy"),
                },
            }

            # Best by dimension
            report["best_by_dimension"] = {}
            for dim in self.study_config.semantic_dims:
                dim_results = [r for r in completed_results if r.config.get("semantic_dim") == dim]
                if dim_results:
                    best_dim = min(dim_results, key=lambda r: r.best_loss)
                    report["best_by_dimension"][str(dim)] = {
                        "experiment_id": best_dim.experiment_id,
                        "best_loss": best_dim.best_loss,
                    }

            # Summary statistics
            all_losses = [r.best_loss for r in completed_results]
            report["loss_statistics"] = {
                "min": min(all_losses),
                "max": max(all_losses),
                "mean": sum(all_losses) / len(all_losses),
                "std": self._std(all_losses),
            }

        # Save report
        path = self.output_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[ReportGenerator] Generated summary report: {path}")
        return str(path)

    def generate_dimension_analysis(self) -> str:
        """
        Generate analysis report by semantic dimension.

        Returns:
            Path to generated report
        """
        report = {
            "study_name": self.study_config.study_name,
            "analysis_type": "dimension_impact",
            "generated_at": datetime.now().isoformat(),
            "dimensions": {},
        }

        completed_results = [r for r in self.results.values() if r.completed]

        for dim in self.study_config.semantic_dims:
            dim_results = [r for r in completed_results if r.config.get("semantic_dim") == dim]

            if not dim_results:
                continue

            losses = [r.best_loss for r in dim_results]
            times = [
                r.resource_summary.total_training_time_seconds
                for r in dim_results
                if r.resource_summary
            ]
            gpu_mem = [
                r.resource_summary.peak_gpu_memory_mb
                for r in dim_results
                if r.resource_summary
            ]

            report["dimensions"][str(dim)] = {
                "num_experiments": len(dim_results),
                "loss": {
                    "min": min(losses),
                    "max": max(losses),
                    "mean": sum(losses) / len(losses),
                    "std": self._std(losses),
                },
            }

            if times:
                report["dimensions"][str(dim)]["training_time_seconds"] = {
                    "min": min(times),
                    "max": max(times),
                    "mean": sum(times) / len(times),
                }

            if gpu_mem:
                report["dimensions"][str(dim)]["gpu_memory_mb"] = {
                    "min": min(gpu_mem),
                    "max": max(gpu_mem),
                    "mean": sum(gpu_mem) / len(gpu_mem),
                }

        # Save report
        path = self.output_dir / "dimension_analysis.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[ReportGenerator] Generated dimension analysis: {path}")
        return str(path)

    def generate_adapter_analysis(self) -> str:
        """
        Generate analysis report by adapter configuration.

        Returns:
            Path to generated report
        """
        report = {
            "study_name": self.study_config.study_name,
            "analysis_type": "adapter_impact",
            "generated_at": datetime.now().isoformat(),
            "by_hidden_size": {},
            "by_hidden_layers": {},
            "heatmap_data": [],
        }

        completed_results = [r for r in self.results.values() if r.completed]

        # Analysis by hidden size
        for hidden_size in self.study_config.hidden_sizes:
            matching = [
                r for r in completed_results
                if r.config.get("encoder_adapter", {}).get("hidden_size") == hidden_size
            ]
            if matching:
                losses = [r.best_loss for r in matching]
                report["by_hidden_size"][str(hidden_size)] = {
                    "num_experiments": len(matching),
                    "mean_loss": sum(losses) / len(losses),
                    "min_loss": min(losses),
                    "std_loss": self._std(losses),
                }

        # Analysis by hidden layers
        for hidden_layers in self.study_config.hidden_layers:
            matching = [
                r for r in completed_results
                if r.config.get("encoder_adapter", {}).get("hidden_layers") == hidden_layers
            ]
            if matching:
                losses = [r.best_loss for r in matching]
                report["by_hidden_layers"][str(hidden_layers)] = {
                    "num_experiments": len(matching),
                    "mean_loss": sum(losses) / len(losses),
                    "min_loss": min(losses),
                    "std_loss": self._std(losses),
                }

        # Heatmap data (hidden_size x hidden_layers)
        for hidden_size in self.study_config.hidden_sizes:
            for hidden_layers in self.study_config.hidden_layers:
                matching = [
                    r for r in completed_results
                    if r.config.get("encoder_adapter", {}).get("hidden_size") == hidden_size
                    and r.config.get("encoder_adapter", {}).get("hidden_layers") == hidden_layers
                ]
                if matching:
                    losses = [r.best_loss for r in matching]
                    report["heatmap_data"].append({
                        "hidden_size": hidden_size,
                        "hidden_layers": hidden_layers,
                        "mean_loss": sum(losses) / len(losses),
                        "num_experiments": len(matching),
                    })

        # Save report
        path = self.output_dir / "adapter_analysis.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[ReportGenerator] Generated adapter analysis: {path}")
        return str(path)

    def export_csv(self) -> str:
        """
        Export all results to CSV for external analysis.

        Returns:
            Path to generated CSV
        """
        path = self.output_dir / "results.csv"

        fieldnames = [
            "experiment_id",
            "semantic_dim",
            "hidden_size",
            "hidden_layers",
            "init_strategy",
            "final_loss",
            "best_loss",
            "best_epoch",
            "total_time_seconds",
            "peak_gpu_mb",
            "avg_epoch_time",
            "completed",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

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
                    row["total_time_seconds"] = result.resource_summary.total_training_time_seconds
                    row["peak_gpu_mb"] = result.resource_summary.peak_gpu_memory_mb
                    row["avg_epoch_time"] = result.resource_summary.avg_epoch_time_seconds

                writer.writerow(row)

        print(f"[ReportGenerator] Exported CSV: {path}")
        return str(path)

    @staticmethod
    def _std(values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
