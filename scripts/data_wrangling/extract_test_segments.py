"""
Extract audio and frames for all segments from 5 test videos.

This script:
1. Loads raw MOSI data to get actual segment IDs
2. Filters to segments from our 5 test videos
3. Extracts audio and frames for each segment
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from Training.Data_Wrangling.mosi_audio_extractor import (
    parse_segment_id,
    get_segment_timestamps,
    extract_audio_segment
)
from Training.Data_Wrangling.mosi_frame_extractor import extract_frame

# Test video IDs
TEST_VIDEO_IDS = [
    'BvYR0L6f2Ig',
    'Vj1wYRQjB-o',
    '9J25DZhivz8',
    'tmZoasNr4rU',
    'Bfr499ggo-0',
]

MOSI_DATA_PATH = 'data/cmumosi/mosi/'
VIDEO_DIR = 'data/cmumosi/videos'
AUDIO_DIR = 'data/cmumosi/audio'
FRAMES_DIR = 'data/cmumosi/frames'


def get_test_segments():
    """Get all segment IDs from raw MOSI data for our 5 test videos."""
    try:
        from mmsdk import mmdatasdk
    except ImportError:
        print("[ERROR] CMU-MultimodalSDK not installed")
        return []

    print("Loading raw MOSI data...")
    raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, MOSI_DATA_PATH)
    words_seq = raw_data.computational_sequences['words']

    # Filter to our test videos
    test_segments = []
    for seg_id in words_seq.data.keys():
        # Check segment ID format (could be 'video_id[num]' or just 'video_id')
        if '[' in seg_id:
            video_id = seg_id.split('[')[0]
        else:
            # Segment ID might be in format 'video_id' without brackets
            # Extract video ID (YouTube video IDs are 11 characters)
            video_id = seg_id[:11] if len(seg_id) >= 11 else seg_id

        if video_id in TEST_VIDEO_IDS:
            test_segments.append(seg_id)

    # Show first few segment IDs to understand format
    print(f"\nFirst few segment IDs (to understand format):")
    for seg_id in sorted(test_segments)[:3]:
        print(f"  {seg_id}")

    return sorted(test_segments)


def main():
    print("=" * 80)
    print("Extracting Audio and Frames for Test Segments")
    print("=" * 80)

    # Create directories
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Get segment IDs from raw MOSI
    print("\n[Step 1/3] Getting segment IDs from raw MOSI data...")
    test_segments = get_test_segments()
    print(f"[OK] Found {len(test_segments)} segments from {len(TEST_VIDEO_IDS)} test videos")

    # Count segments per video
    video_counts = {}
    for seg_id in test_segments:
        video_id = seg_id.split('[')[0]
        video_counts[video_id] = video_counts.get(video_id, 0) + 1

    print("\nSegment breakdown:")
    for video_id in TEST_VIDEO_IDS:
        count = video_counts.get(video_id, 0)
        print(f"  {video_id}: {count} segments")

    # Extract audio
    print(f"\n[Step 2/3] Extracting audio for {len(test_segments)} segments...")
    print("-" * 80)

    audio_success = 0
    audio_failed = 0

    for seg_id in test_segments:
        # Parse segment ID (handle different formats)
        if '[' in seg_id:
            video_id, segment_num = parse_segment_id(seg_id)
        else:
            # Segment ID without brackets - use as-is
            video_id = seg_id[:11] if len(seg_id) >= 11 else seg_id
            segment_num = 0  # Default segment number

        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        audio_path = os.path.join(AUDIO_DIR, f"{seg_id.replace('[', '_').replace(']', '')}.wav")

        # Skip if already exists
        if os.path.exists(audio_path):
            print(f"[SKIP] {seg_id} - audio already exists")
            audio_success += 1
            continue

        # Check video exists
        if not os.path.exists(video_path):
            print(f"[FAIL] {seg_id} - video not found")
            audio_failed += 1
            continue

        try:
            start_time, end_time = get_segment_timestamps(seg_id, MOSI_DATA_PATH)
            result = extract_audio_segment(video_path, start_time, end_time, audio_path)

            if result['status'] == 'success':
                print(f"[OK] {seg_id} - extracted ({result['duration']:.2f}s)")
                audio_success += 1
            else:
                print(f"[FAIL] {seg_id} - {result.get('error', 'unknown error')}")
                audio_failed += 1

        except Exception as e:
            print(f"[ERROR] {seg_id} - {str(e)[:100]}")
            audio_failed += 1

    print(f"\nAudio extraction: {audio_success}/{len(test_segments)} successful")

    # Extract frames
    print(f"\n[Step 3/3] Extracting frames for {len(test_segments)} segments...")
    print("-" * 80)

    frame_success = 0
    frame_failed = 0

    for seg_id in test_segments:
        # Parse segment ID (handle different formats)
        if '[' in seg_id:
            video_id, segment_num = parse_segment_id(seg_id)
        else:
            # Segment ID without brackets - use as-is
            video_id = seg_id[:11] if len(seg_id) >= 11 else seg_id
            segment_num = 0  # Default segment number

        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        frame_path = os.path.join(FRAMES_DIR, f"{seg_id.replace('[', '_').replace(']', '')}.jpg")

        # Skip if already exists
        if os.path.exists(frame_path):
            print(f"[SKIP] {seg_id} - frame already exists")
            frame_success += 1
            continue

        # Check video exists
        if not os.path.exists(video_path):
            print(f"[FAIL] {seg_id} - video not found")
            frame_failed += 1
            continue

        try:
            start_time, end_time = get_segment_timestamps(seg_id, MOSI_DATA_PATH)
            result = extract_frame(video_path, start_time, end_time, frame_path, strategy='middle')

            if result['status'] == 'success':
                print(f"[OK] {seg_id} - extracted ({result['resolution']})")
                frame_success += 1
            else:
                print(f"[FAIL] {seg_id} - {result.get('error', 'unknown error')}")
                frame_failed += 1

        except Exception as e:
            print(f"[ERROR] {seg_id} - {str(e)[:100]}")
            frame_failed += 1

    print(f"\nFrame extraction: {frame_success}/{len(test_segments)} successful")

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total segments from raw MOSI: {len(test_segments)}")
    print(f"Audio extractions: {audio_success}/{len(test_segments)}")
    print(f"Frame extractions: {frame_success}/{len(test_segments)}")
    print()

    if audio_success > 0 and frame_success > 0:
        print("[OK] Extraction successful! Ready to test dataset loader.")
        print("\nNext step: Test MOSIRawVideoDataset")
    else:
        print("[WARNING] Extraction incomplete. Check errors above.")

    print("=" * 80)


if __name__ == "__main__":
    main()
