#!/usr/bin/env python3
"""
CLI Entry Point for Factorization Study

Runs automated hyperparameter sweep to study performance impacts of:
- Semantic vector dimensions (256, 512, 1024, 2048)
- Encoder adapter configurations (hidden size, layers)
- Initialization strategies (pretrained, random)

Usage:
    # Full sweep (~60 experiments)
    python scripts/run_factorization_study.py --preset full_sweep

    # Quick test (4 experiments)
    python scripts/run_factorization_study.py --preset quick

    # Standard sweep (16 experiments)
    python scripts/run_factorization_study.py --preset standard

    # Resume interrupted study
    python scripts/run_factorization_study.py --resume studies/my_study

    # Custom configuration
    python scripts/run_factorization_study.py --config study_config.yaml

    # List what would run without executing
    python scripts/run_factorization_study.py --preset quick --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.FactorizationStudy.config.study_config import (
    StudyConfig,
    StudyPreset,
    create_study_config,
)
from src.FactorizationStudy.config.config_generator import ConfigGenerator
from src.FactorizationStudy.runner.study_runner import StudyRunner
from src.FactorizationStudy.reporting.report_generator import ReportGenerator
from src.FactorizationStudy.reporting.visualization import Visualizer


# Data configuration
MOSI_DATA_PATH = os.getenv('MOSI_DATA_PATH', 'data/cmumosi/mosi/')
AUDIO_DIR = os.getenv('AUDIO_DIR', 'data/cmumosi/audio/')
VIDEO_DIR = os.getenv('VIDEO_DIR', 'data/cmumosi/frames/')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Factorization Study for Semantic Vector Space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Preset selection
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "standard", "full_sweep"],
        default="standard",
        help="Study preset: quick (4 exp), standard (16 exp), full_sweep (~60 exp)",
    )

    # Study name
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Study name (default: auto-generated with timestamp)",
    )

    # Resume from existing study
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing study directory to resume",
    )

    # Custom config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom study configuration YAML/JSON file",
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default="studies",
        help="Output directory for study results (default: studies/)",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override default epochs per experiment",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override default learning rate",
    )

    # Limit experiments
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Limit number of experiments to run",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing",
    )

    # Skip reporting
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip generating reports and visualizations",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def setup_data_loaders(batch_size: int = 32):
    """
    Set up training and validation data loaders.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
    from src.Training.Data_Wrangling.collate_functions import collate_fn_raw_video

    print(f"\n[Setup] Loading MOSI dataset...")
    print(f"  MOSI data: {MOSI_DATA_PATH}")
    print(f"  Audio dir: {AUDIO_DIR}")
    print(f"  Video dir: {VIDEO_DIR}")

    try:
        train_dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path=MOSI_DATA_PATH,
            audio_dir=AUDIO_DIR,
            video_dir=VIDEO_DIR,
        )

        val_dataset = MOSIRawVideoDataset(
            split='val',
            mosi_data_path=MOSI_DATA_PATH,
            audio_dir=AUDIO_DIR,
            video_dir=VIDEO_DIR,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_raw_video,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_raw_video,
            num_workers=0,
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        return train_loader, val_loader

    except Exception as e:
        print(f"  [WARNING] Could not load MOSI dataset: {e}")
        print(f"  [WARNING] Ensure data is downloaded and paths are correct")
        raise


def create_study_from_args(args) -> StudyConfig:
    """Create study configuration from command line arguments."""

    # Generate study name if not provided
    if args.name:
        study_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{args.preset}_{timestamp}"

    # Map preset string to enum
    preset_map = {
        "quick": StudyPreset.QUICK,
        "standard": StudyPreset.STANDARD,
        "full_sweep": StudyPreset.FULL_SWEEP,
    }
    preset = preset_map.get(args.preset, StudyPreset.STANDARD)

    # Create config
    overrides = {"output_dir": args.output_dir}
    if args.epochs:
        overrides["default_epochs"] = args.epochs
    if args.batch_size:
        overrides["default_batch_size"] = args.batch_size
    if args.learning_rate:
        overrides["default_learning_rate"] = args.learning_rate

    config = create_study_config(study_name, preset=preset, **overrides)

    return config


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("FACTORIZATION STUDY - Semantic Vector Space Architecture")
    print("=" * 70)

    # Display configuration
    print(f"\nDevice: {DEVICE}")
    print(f"Preset: {args.preset}")

    # Resume or create new study
    if args.resume:
        print(f"\nResuming study from: {args.resume}")
        config_path = Path(args.resume) / "study_config.json"
        if not config_path.exists():
            print(f"[ERROR] No study_config.json found in {args.resume}")
            sys.exit(1)
        study_config = StudyConfig.load(str(config_path))
    elif args.config:
        print(f"\nLoading custom config from: {args.config}")
        study_config = StudyConfig.load(args.config)
    else:
        study_config = create_study_from_args(args)

    # Generate experiments
    generator = ConfigGenerator(study_config)
    generator.generate()

    # Display study info
    print(f"\nStudy: {study_config.study_name}")
    print(f"Output: {study_config.get_study_dir()}")
    print(f"\n{generator.summary()}")

    # Dry run - just show what would run
    if args.dry_run:
        print("\n[DRY RUN] Would run the following experiments:")
        for i, exp in enumerate(study_config.experiments[:10], 1):
            print(f"  {i}. {exp.experiment_id}")
        if len(study_config.experiments) > 10:
            print(f"  ... and {len(study_config.experiments) - 10} more")
        print("\n[DRY RUN] No experiments executed.")
        return

    # Set up data loaders
    batch_size = args.batch_size or study_config.default_batch_size
    try:
        train_loader, val_loader = setup_data_loaders(batch_size)
    except Exception as e:
        print(f"\n[ERROR] Failed to set up data loaders: {e}")
        print("[ERROR] Please ensure MOSI data is downloaded and preprocessed.")
        sys.exit(1)

    # Run study
    print("\nStarting experiments...")
    runner = StudyRunner(study_config, train_loader, val_loader)
    results = runner.run(max_experiments=args.max_experiments)

    # Generate reports
    if not args.skip_reports and results:
        print("\nGenerating reports...")
        report_gen = ReportGenerator(study_config, results)
        report_gen.generate_all_reports()

        visualizer = Visualizer(study_config, results)
        visualizer.generate_all_plots()

    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {study_config.get_study_dir()}")
    print(f"Reports: {study_config.get_study_dir() / 'reports'}")


if __name__ == "__main__":
    main()
