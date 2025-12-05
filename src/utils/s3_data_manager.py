"""
S3 Data Manager for CS627 Training Pipeline

Handles downloading training data from S3 and uploading/downloading weights.
Supports both IAM roles (EC2) and explicit credentials (local dev).

Environment Variables:
    S3_BUCKET: S3 bucket name (required)
    S3_REGION: AWS region (default: us-east-1)
    USE_S3: Enable S3 data loading (default: 0)
    S3_DATA_PREFIX: S3 prefix for training data (default: data/cmumosi/)
    S3_WEIGHTS_PREFIX: S3 prefix for weights (default: OptimalWeights/)

Usage:
    from src.utils.s3_data_manager import download_training_data, sync_optimal_weights

    # Download training data from S3 to local disk
    download_training_data(bucket='cs627-svs-data', local_path='data/cmumosi/')

    # Sync OptimalWeights from S3
    sync_optimal_weights(bucket='cs627-svs-data', direction='download')
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def get_s3_client(region: Optional[str] = None) -> boto3.client:
    """
    Create S3 client. Uses IAM role on EC2 or credentials from env/config.

    Args:
        region: AWS region (default: from S3_REGION env or us-east-1)

    Returns:
        boto3 S3 client
    """
    region = region or os.getenv('S3_REGION', 'us-east-1')
    return boto3.client('s3', region_name=region)


def check_s3_connection(bucket: str) -> bool:
    """
    Verify S3 bucket is accessible.

    Args:
        bucket: S3 bucket name

    Returns:
        True if bucket is accessible, False otherwise
    """
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=bucket)
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"[S3] Connection failed: {e}")
        return False


def download_training_data(
    bucket: str,
    local_path: str = "data/cmumosi/",
    s3_prefix: str = None,
    force: bool = False
) -> bool:
    """
    Download training data from S3 to local disk.

    Skips download if local data already exists (unless force=True).
    Shows progress bar during download.

    Args:
        bucket: S3 bucket name
        local_path: Local directory to download to
        s3_prefix: S3 prefix (default: from S3_DATA_PREFIX env or data/cmumosi/)
        force: Force re-download even if local data exists

    Returns:
        True if download successful or data already exists, False on error
    """
    s3_prefix = s3_prefix or os.getenv('S3_DATA_PREFIX', 'data/cmumosi/')

    # Normalize paths
    local_path = Path(local_path)
    s3_prefix = s3_prefix.rstrip('/') + '/'

    # Check if local data exists (skip if already present)
    marker_files = ['mosi/CMU_MOSI_TimestampedWords.cdf', 'audio', 'frames']
    data_exists = all((local_path / f).exists() for f in marker_files[:1])

    if data_exists and not force:
        print(f"[S3] Training data already exists at {local_path}")
        print(f"[S3] Skipping download (use force=True to re-download)")
        return True

    print(f"[S3] Downloading training data from s3://{bucket}/{s3_prefix}")
    print(f"[S3] Destination: {local_path}")

    try:
        s3 = get_s3_client()

        # List all objects in the prefix
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

        # Collect all objects
        objects = []
        total_size = 0
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects.append(obj)
                    total_size += obj['Size']

        if not objects:
            print(f"[S3] No objects found at s3://{bucket}/{s3_prefix}")
            return False

        print(f"[S3] Found {len(objects)} files ({total_size / 1024 / 1024:.1f} MB)")

        # Download with progress bar
        downloaded_size = 0
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for obj in objects:
                key = obj['Key']
                # Remove prefix to get relative path
                relative_path = key[len(s3_prefix):]
                if not relative_path:
                    continue

                local_file = local_path / relative_path

                # Create parent directories
                local_file.parent.mkdir(parents=True, exist_ok=True)

                # Download file
                s3.download_file(bucket, key, str(local_file))

                pbar.update(obj['Size'])
                downloaded_size += obj['Size']

        print(f"[S3] Download complete: {downloaded_size / 1024 / 1024:.1f} MB")
        return True

    except ClientError as e:
        print(f"[S3] Download failed: {e}")
        return False
    except NoCredentialsError:
        print("[S3] No AWS credentials found. Configure via:")
        print("  - IAM role (on EC2)")
        print("  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("  - AWS CLI config (~/.aws/credentials)")
        return False


def upload_weights(
    bucket: str,
    local_path: str,
    s3_prefix: str
) -> bool:
    """
    Upload weights directory to S3.

    Args:
        bucket: S3 bucket name
        local_path: Local directory containing weights
        s3_prefix: S3 prefix to upload to

    Returns:
        True if upload successful, False on error
    """
    local_path = Path(local_path)
    s3_prefix = s3_prefix.rstrip('/') + '/'

    if not local_path.exists():
        print(f"[S3] Local path does not exist: {local_path}")
        return False

    print(f"[S3] Uploading weights from {local_path}")
    print(f"[S3] Destination: s3://{bucket}/{s3_prefix}")

    try:
        s3 = get_s3_client()

        # Find all files to upload
        files = list(local_path.rglob('*'))
        files = [f for f in files if f.is_file()]

        if not files:
            print(f"[S3] No files found in {local_path}")
            return False

        total_size = sum(f.stat().st_size for f in files)
        print(f"[S3] Uploading {len(files)} files ({total_size / 1024 / 1024:.1f} MB)")

        # Upload with progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Uploading") as pbar:
            for file in files:
                relative_path = file.relative_to(local_path)
                s3_key = s3_prefix + str(relative_path).replace('\\', '/')

                s3.upload_file(str(file), bucket, s3_key)
                pbar.update(file.stat().st_size)

        print(f"[S3] Upload complete")
        return True

    except ClientError as e:
        print(f"[S3] Upload failed: {e}")
        return False
    except NoCredentialsError:
        print("[S3] No AWS credentials found")
        return False


def sync_optimal_weights(
    bucket: str,
    direction: str = "download",
    local_path: str = "OptimalWeights/",
    s3_prefix: str = None
) -> bool:
    """
    Sync OptimalWeights between local and S3.

    Args:
        bucket: S3 bucket name
        direction: "download" or "upload"
        local_path: Local OptimalWeights directory
        s3_prefix: S3 prefix (default: from S3_WEIGHTS_PREFIX env or OptimalWeights/)

    Returns:
        True if sync successful, False on error
    """
    s3_prefix = s3_prefix or os.getenv('S3_WEIGHTS_PREFIX', 'OptimalWeights/')
    local_path = Path(local_path)

    if direction == "download":
        print(f"[S3] Syncing OptimalWeights from S3...")
        return download_training_data(
            bucket=bucket,
            local_path=str(local_path),
            s3_prefix=s3_prefix,
            force=True  # Always sync weights
        )
    elif direction == "upload":
        print(f"[S3] Syncing OptimalWeights to S3...")
        return upload_weights(
            bucket=bucket,
            local_path=str(local_path),
            s3_prefix=s3_prefix
        )
    else:
        print(f"[S3] Invalid direction: {direction}. Use 'download' or 'upload'")
        return False


def download_if_enabled() -> bool:
    """
    Download training data from S3 if USE_S3=1.

    Convenience function for use in training scripts.

    Returns:
        True if S3 disabled or download successful, False on error
    """
    if os.getenv('USE_S3', '0') != '1':
        return True

    bucket = os.getenv('S3_BUCKET')
    if not bucket:
        print("[S3] USE_S3=1 but S3_BUCKET not set")
        return False

    # Download training data
    mosi_path = os.getenv('MOSI_DATA_PATH', 'data/cmumosi/mosi/')
    audio_path = os.getenv('AUDIO_DIR', 'data/cmumosi/audio/')
    video_path = os.getenv('VIDEO_DIR', 'data/cmumosi/frames/')

    # Download to parent directory (data/cmumosi/)
    parent_path = Path(mosi_path).parent
    return download_training_data(bucket=bucket, local_path=str(parent_path))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='S3 Data Manager for CS627')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--action', choices=['download', 'upload', 'check'],
                        default='download', help='Action to perform')
    parser.add_argument('--local-path', default='data/cmumosi/',
                        help='Local path for data')
    parser.add_argument('--s3-prefix', default='data/cmumosi/',
                        help='S3 prefix')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download')

    args = parser.parse_args()

    if args.action == 'check':
        if check_s3_connection(args.bucket):
            print(f"[OK] S3 bucket {args.bucket} is accessible")
        else:
            print(f"[ERROR] Cannot access S3 bucket {args.bucket}")
    elif args.action == 'download':
        download_training_data(
            bucket=args.bucket,
            local_path=args.local_path,
            s3_prefix=args.s3_prefix,
            force=args.force
        )
    elif args.action == 'upload':
        upload_weights(
            bucket=args.bucket,
            local_path=args.local_path,
            s3_prefix=args.s3_prefix
        )
