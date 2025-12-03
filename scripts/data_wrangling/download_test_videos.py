"""
Download 5 test videos from MOSI dataset for pipeline validation.

Uses the video IDs that we already confirmed are available from test_mosi_video_access.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from Training.Data_Wrangling.mosi_video_downloader import download_single_video

# These video IDs were confirmed available in our earlier test
TEST_VIDEO_IDS = [
    'BvYR0L6f2Ig',  # Video 1 - Available
    'Vj1wYRQjB-o',  # Video 2 - Available
    '9J25DZhivz8',  # Video 3 - Available
    'tmZoasNr4rU',  # Video 4 - Available
    'Bfr499ggo-0',  # Video 5 - Available
]

OUTPUT_DIR = 'data/cmumosi/videos'


def main():
    print("=" * 80)
    print("Downloading 5 Test Videos for Pipeline Validation")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    successful = 0

    for i, video_id in enumerate(TEST_VIDEO_IDS, 1):
        print(f"\n[{i}/5] Downloading {video_id}...")
        result = download_single_video(video_id, OUTPUT_DIR)
        results.append(result)

        if result['status'] in ['success', 'already_exists']:
            successful += 1
            status_icon = "[OK]"
            size_info = f"{result.get('size_mb', 0):.1f}MB"
        else:
            status_icon = "[FAIL]"
            size_info = result.get('error', 'Unknown error')[:50]

        print(f"{status_icon} {video_id} - {result['status']} - {size_info}")

    print("\n" + "=" * 80)
    print(f"Downloaded: {successful}/5 videos")
    print("=" * 80)

    if successful >= 3:
        print("\n[OK] Sufficient videos for testing!")
        print("Next steps:")
        print("  1. Extract audio segments")
        print("  2. Extract video frames")
        print("  3. Test full pipeline")
    else:
        print("\n[WARNING] Only {successful} videos downloaded")
        print("May need to select different test videos")

    return results


if __name__ == "__main__":
    main()
