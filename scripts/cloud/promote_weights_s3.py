#!/usr/bin/env python3
"""
Promote trained weights to/from S3 OptimalWeights.

This script handles syncing weights between local CandidateWeights and S3 storage.
Use after evaluating candidate weights to make them available for future training runs.

Usage:
    # Promote local CandidateWeights to S3 OptimalWeights
    python scripts/cloud/promote_weights_s3.py --source CandidateWeights/docker_20241205_123456/ --bucket cs627-svs-data

    # Download OptimalWeights from S3 to local
    python scripts/cloud/promote_weights_s3.py --download --bucket cs627-svs-data

    # List available runs in CandidateWeights
    python scripts/cloud/promote_weights_s3.py --list

    # Promote best run automatically
    python scripts/cloud/promote_weights_s3.py --best --bucket cs627-svs-data

Environment Variables:
    S3_BUCKET: Default S3 bucket name
    S3_REGION: AWS region (default: us-east-1)
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.s3_data_manager import upload_weights, sync_optimal_weights, check_s3_connection

# Import compare_weights for run discovery
from scripts.cloud.compare_weights import get_training_runs, find_best_run, list_runs

CANDIDATE_WEIGHTS_DIR = PROJECT_ROOT / "CandidateWeights"
OPTIMAL_WEIGHTS_DIR = PROJECT_ROOT / "OptimalWeights"


def promote_to_s3(source_path: str, bucket: str, force: bool = False) -> bool:
    """
    Promote candidate weights to S3 OptimalWeights.

    Args:
        source_path: Path to CandidateWeights run directory
        bucket: S3 bucket name
        force: Skip confirmation

    Returns:
        True if successful, False otherwise
    """
    source_path = Path(source_path)

    # Check if source exists
    if not source_path.exists():
        print(f"[ERROR] Source path does not exist: {source_path}")
        return False

    # Find weight files
    weights_dir = source_path / "OptimalWeights"
    if not weights_dir.exists():
        # Try source_path directly if it's the weights directory
        if list(source_path.glob("*.pth")):
            weights_dir = source_path
        else:
            print(f"[ERROR] No OptimalWeights directory found in {source_path}")
            return False

    weight_files = list(weights_dir.glob("*.pth"))
    if not weight_files:
        print(f"[ERROR] No .pth weight files found in {weights_dir}")
        return False

    print("=" * 80)
    print("Promote Weights to S3")
    print("=" * 80)
    print()
    print(f"Source: {weights_dir}")
    print(f"Target: s3://{bucket}/OptimalWeights/")
    print()
    print("Files to upload:")
    total_size = 0
    for wf in weight_files:
        size = wf.stat().st_size
        total_size += size
        print(f"  - {wf.name} ({size / 1024 / 1024:.1f} MB)")
    print(f"\nTotal: {len(weight_files)} files ({total_size / 1024 / 1024:.1f} MB)")
    print()

    if not force:
        response = input("Proceed with upload to S3? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return False

    # Upload to S3
    success = upload_weights(
        bucket=bucket,
        local_path=str(weights_dir),
        s3_prefix="OptimalWeights/"
    )

    if success:
        print()
        print("[OK] Weights promoted to S3 successfully!")
        print()
        print("To use these weights on EC2:")
        print("  1. Weights will auto-download with USE_S3=1")
        print("  2. Or manually: aws s3 sync s3://{bucket}/OptimalWeights/ OptimalWeights/")
        print()
        print("To sync to local machine:")
        print(f"  python scripts/cloud/promote_weights_s3.py --download --bucket {bucket}")

    return success


def download_from_s3(bucket: str, local_path: str = None) -> bool:
    """
    Download OptimalWeights from S3 to local.

    Args:
        bucket: S3 bucket name
        local_path: Local OptimalWeights directory (default: OptimalWeights/)

    Returns:
        True if successful, False otherwise
    """
    local_path = local_path or str(OPTIMAL_WEIGHTS_DIR)

    print("=" * 80)
    print("Download Weights from S3")
    print("=" * 80)
    print()
    print(f"Source: s3://{bucket}/OptimalWeights/")
    print(f"Target: {local_path}")
    print()

    success = sync_optimal_weights(
        bucket=bucket,
        direction="download",
        local_path=local_path
    )

    if success:
        print()
        print("[OK] Weights downloaded successfully!")
        # List downloaded files
        downloaded = list(Path(local_path).glob("*.pth"))
        if downloaded:
            print(f"\nDownloaded {len(downloaded)} weight files:")
            for f in downloaded:
                print(f"  - {f.name}")

    return success


def promote_best_to_s3(bucket: str, force: bool = False) -> bool:
    """
    Find and promote the best run to S3.

    Args:
        bucket: S3 bucket name
        force: Skip confirmation

    Returns:
        True if successful, False otherwise
    """
    runs = get_training_runs()
    best = find_best_run(runs)

    if not best:
        print("[ERROR] No runs with metrics found.")
        print("Ensure training_metrics.json is present in training_reports/")
        return False

    print("=" * 80)
    print("Best Training Run Found")
    print("=" * 80)
    print()
    print(f"Run: {best['name']}")
    if best['timestamp']:
        print(f"Timestamp: {best['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    if best['metrics'] and 'final_loss' in best['metrics']:
        print(f"Final Loss: {best['metrics']['final_loss']:.4f}")
    print(f"Weight Files: {len(best['weight_files'])}")
    print()

    return promote_to_s3(str(best['path']), bucket, force)


def main():
    parser = argparse.ArgumentParser(
        description="Promote trained weights to/from S3 OptimalWeights"
    )

    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        '--source', metavar='PATH',
        help='Promote weights from this CandidateWeights directory to S3'
    )
    action_group.add_argument(
        '--download', action='store_true',
        help='Download OptimalWeights from S3 to local'
    )
    action_group.add_argument(
        '--best', action='store_true',
        help='Automatically find and promote the best run to S3'
    )
    action_group.add_argument(
        '--list', action='store_true',
        help='List available runs in CandidateWeights'
    )
    action_group.add_argument(
        '--check', action='store_true',
        help='Check S3 connection'
    )

    # Common arguments
    parser.add_argument(
        '--bucket', '-b',
        default=os.getenv('S3_BUCKET', 'cs627-svs-data'),
        help='S3 bucket name (default: $S3_BUCKET or cs627-svs-data)'
    )
    parser.add_argument(
        '--local-path',
        default=str(OPTIMAL_WEIGHTS_DIR),
        help='Local OptimalWeights directory for download'
    )
    parser.add_argument(
        '--force', '-f', action='store_true',
        help='Skip confirmation prompts'
    )

    args = parser.parse_args()

    # Handle actions
    if args.list:
        runs = get_training_runs()
        list_runs(runs)
        return

    if args.check:
        print(f"Checking connection to s3://{args.bucket}...")
        if check_s3_connection(args.bucket):
            print(f"[OK] S3 bucket {args.bucket} is accessible")
        else:
            print(f"[ERROR] Cannot access S3 bucket {args.bucket}")
            sys.exit(1)
        return

    if args.download:
        success = download_from_s3(args.bucket, args.local_path)
        sys.exit(0 if success else 1)

    if args.best:
        success = promote_best_to_s3(args.bucket, args.force)
        sys.exit(0 if success else 1)

    if args.source:
        success = promote_to_s3(args.source, args.bucket, args.force)
        sys.exit(0 if success else 1)

    # No action specified - show help
    parser.print_help()
    print()
    print("Examples:")
    print("  # Upload best run to S3")
    print("  python scripts/cloud/promote_weights_s3.py --best --bucket cs627-svs-data")
    print()
    print("  # Upload specific run to S3")
    print("  python scripts/cloud/promote_weights_s3.py --source CandidateWeights/docker_20241205_123456/ --bucket cs627-svs-data")
    print()
    print("  # Download from S3 to local")
    print("  python scripts/cloud/promote_weights_s3.py --download --bucket cs627-svs-data")


if __name__ == "__main__":
    main()
