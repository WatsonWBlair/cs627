"""
Test script to validate YouTube video availability for MOSI dataset.

This CRITICAL test determines if the raw video extraction approach is viable.
If <50% of videos are available, we'll need to adjust our strategy.
"""

import pickle
import os
import sys

# Check for yt-dlp
try:
    import yt_dlp
except ImportError:
    print("[ERROR] yt-dlp not installed")
    print("Install with: pip install yt-dlp")
    sys.exit(1)


def parse_segment_id(seg_id):
    """Extract video ID from segment ID.

    Example: 'phqU0CNn4R0[2]' → 'phqU0CNn4R0'
    """
    return seg_id.split('[')[0]


def test_video_availability(segment_ids, num_samples=10):
    """
    Test if MOSI YouTube videos are still available.

    Args:
        segment_ids: List of MOSI segment IDs
        num_samples: Number of samples to test

    Returns:
        Success rate (0.0 to 1.0)
    """
    # Get unique video IDs from segments
    sample_seg_ids = segment_ids[:num_samples]
    video_ids = [parse_segment_id(sid) for sid in sample_seg_ids]
    unique_video_ids = list(set(video_ids))

    print(f"\nTesting {len(unique_video_ids)} unique videos from {num_samples} segments...")
    print("=" * 80)

    success_count = 0
    results = []

    for i, video_id in enumerate(unique_video_ids, 1):
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        # Configure yt-dlp to just check info, not download
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)

                if info:
                    print(f"[{i}/{len(unique_video_ids)}] [OK] {video_id} - Available")
                    success_count += 1
                    results.append({'video_id': video_id, 'status': 'available', 'title': info.get('title', 'Unknown')})
                else:
                    print(f"[{i}/{len(unique_video_ids)}] [FAIL] {video_id} - Info extraction failed")
                    results.append({'video_id': video_id, 'status': 'failed', 'error': 'No info'})

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            if 'Private video' in error_msg:
                status = 'private'
            elif 'Video unavailable' in error_msg:
                status = 'unavailable'
            elif 'This video has been removed' in error_msg:
                status = 'removed'
            else:
                status = 'error'

            print(f"[{i}/{len(unique_video_ids)}] [FAIL] {video_id} - {status}")
            results.append({'video_id': video_id, 'status': status, 'error': error_msg[:100]})

        except Exception as e:
            print(f"[{i}/{len(unique_video_ids)}] [ERROR] {video_id} - Unexpected error: {e}")
            results.append({'video_id': video_id, 'status': 'error', 'error': str(e)[:100]})

    success_rate = success_count / len(unique_video_ids)

    print("=" * 80)
    print(f"\nResults: {success_count}/{len(unique_video_ids)} videos available")
    print(f"Success Rate: {success_rate:.1%}")

    # Failure breakdown
    failure_types = {}
    for result in results:
        if result['status'] != 'available':
            failure_types[result['status']] = failure_types.get(result['status'], 0) + 1

    if failure_types:
        print("\nFailure Breakdown:")
        for status, count in failure_types.items():
            print(f"  {status}: {count}")

    return success_rate, results


def main():
    """Main test function."""
    print("=" * 80)
    print("MOSI Video Availability Test")
    print("=" * 80)

    # Load MOSI preprocessed data
    preprocessed_path = 'data/cmumosi/mosi/preprocessed_train.pkl'

    if not os.path.exists(preprocessed_path):
        print(f"[ERROR] Preprocessed data not found at: {preprocessed_path}")
        print("Please ensure MOSI dataset is downloaded and preprocessed.")
        return

    print(f"\nLoading segment IDs from: {preprocessed_path}")

    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
        segment_ids = data['segment_ids']

    print(f"Total segments in train split: {len(segment_ids)}")

    # Test with 10 samples
    success_rate, results = test_video_availability(segment_ids, num_samples=10)

    # Decision guidance
    print("\n" + "=" * 80)
    print("DECISION GUIDANCE")
    print("=" * 80)

    if success_rate >= 0.6:
        print("[OK] Success rate is acceptable!")
        print(f"Expected ~{int(len(segment_ids) * success_rate)} usable segments from {len(segment_ids)} total")
        print("\nRecommendation: PROCEED with video extraction approach")
        print("Next steps:")
        print("  1. mkdir -p data/cmumosi/videos data/cmumosi/audio data/cmumosi/frames")
        print("  2. Implement mosi_video_downloader.py")
        print("  3. Download full dataset")
    elif success_rate >= 0.3:
        print("[WARNING] Success rate is marginal")
        print(f"Expected only ~{int(len(segment_ids) * success_rate)} usable segments from {len(segment_ids)} total")
        print("\nRecommendation: PROCEED but supplement with custom data")
        print("  - Download available MOSI videos")
        print("  - Record additional webcam samples for missing data")
    else:
        print("[CRITICAL] Success rate too low!")
        print(f"Only {int(len(segment_ids) * success_rate)} segments would be usable")
        print("\nRecommendation: PIVOT to alternative approach")
        print("Options:")
        print("  1. Use different dataset with available raw videos")
        print("  2. Create custom dataset from scratch (webcam recordings)")
        print("  3. Use pre-extracted features temporarily (accept limitations)")

    print("=" * 80)


if __name__ == "__main__":
    main()
