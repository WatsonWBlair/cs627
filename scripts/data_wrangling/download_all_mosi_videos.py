"""
Download ALL videos from the CMU-MOSI dataset.

This script:
1. Downloads MOSI metadata (if not already downloaded)
2. Extracts all unique video IDs from the dataset
3. Downloads all available videos in parallel
4. Saves a manifest of successful/failed downloads

Expected: ~2,199 segments from ~93 unique videos
Success rate: 55-60% based on YouTube availability
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from Training.Data_Wrangling.mosi_dataset import download_mosi
from Training.Data_Wrangling.mosi_video_downloader import download_mosi_videos

MOSI_DATA_PATH = 'data/cmumosi/mosi/'
VIDEO_OUTPUT_DIR = 'data/cmumosi/videos'


def get_all_segment_ids():
    """Get all segment IDs from MOSI dataset."""
    try:
        from mmsdk import mmdatasdk
    except ImportError:
        print("[ERROR] CMU-MultimodalSDK not installed!")
        print("Install with: pip install CMU-MultimodalSDK")
        return []

    print("Loading MOSI metadata...")
    raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, MOSI_DATA_PATH)

    # Get segment IDs from words sequence
    words_seq = raw_data.computational_sequences['words']
    segment_ids = list(words_seq.data.keys())

    print(f"Found {len(segment_ids)} segments in MOSI dataset")

    return segment_ids


def main():
    print("=" * 80)
    print("CMU-MOSI Full Dataset Video Download")
    print("=" * 80)
    print()

    # Step 1: Download MOSI metadata
    print("[Step 1/3] Downloading MOSI metadata...")
    os.makedirs(MOSI_DATA_PATH, exist_ok=True)
    download_mosi(MOSI_DATA_PATH)
    print("[OK] MOSI metadata downloaded")
    print()

    # Step 2: Get all segment IDs
    print("[Step 2/3] Extracting segment IDs...")
    segment_ids = get_all_segment_ids()

    if not segment_ids:
        print("[ERROR] No segment IDs found!")
        return

    # Count unique videos
    unique_videos = set(seg_id.split('[')[0] for seg_id in segment_ids)
    print(f"[OK] Found {len(unique_videos)} unique videos from {len(segment_ids)} segments")
    print()

    # Step 3: Download all videos
    print("[Step 3/3] Downloading videos...")
    print(f"This will download ~{len(unique_videos)} videos (may take 1-3 hours)")
    print("Expected success rate: 55-60% (~{} videos)".format(int(len(unique_videos) * 0.575)))
    print()

    # Ask for confirmation
    response = input("Continue with download? (y/N): ")
    if response.lower() != 'y':
        print("[CANCELLED] Download cancelled by user")
        return

    print()
    print("Starting parallel download (4 workers)...")
    print("This will take a while - grab a coffee!")
    print()

    # Download videos with manifest
    manifest = download_mosi_videos(
        segment_ids=segment_ids,
        output_dir=VIDEO_OUTPUT_DIR,
        max_workers=4,
        save_manifest=True
    )

    # Print summary
    print()
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print()
    # Calculate success rate and total size
    total_successful = manifest['successful_downloads'] + manifest['already_exists']
    success_rate = (total_successful / manifest['unique_videos'] * 100) if manifest['unique_videos'] > 0 else 0

    total_size_mb = sum(
        video.get('size_mb', 0)
        for video in manifest['videos'].values()
        if video.get('status') in ['success', 'already_exists']
    )

    print("Summary:")
    print(f"  Total segments:     {manifest['total_segments']}")
    print(f"  Unique videos:      {manifest['unique_videos']}")
    print(f"  Downloaded:         {manifest['successful_downloads']}")
    print(f"  Already existed:    {manifest['already_exists']}")
    print(f"  Failed:             {manifest['failed_downloads']}")
    print(f"  Success rate:       {success_rate:.1f}%")
    print()
    print(f"  Total size:         {total_size_mb:.1f} MB")
    print()
    print("Manifest saved to:")
    print(f"  {VIDEO_OUTPUT_DIR}/download_manifest.json")
    print()
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Extract audio segments: python scripts/data_wrangling/extract_all_segments.py")
    print("  2. Extract video frames:   (same script handles both)")
    print("  3. Train encoders:         python src/Training/train_raw_encoders.py")
    print()


if __name__ == "__main__":
    main()
