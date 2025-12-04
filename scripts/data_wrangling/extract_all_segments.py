"""
Extract audio and frames for ALL segments from downloaded MOSI videos.

This script:
1. Loads download_manifest.json to get available videos
2. Filters MOSI segments to those from downloaded videos
3. Extracts audio (16kHz WAV) and frames (JPG) for each segment

Phase 1: Simple extraction with basic error handling
Phase 2 (future): Add split-aware organization, resume capability, validation
"""

import sys
import os
import json
import pickle
from typing import List

# Add src to path
sys.path.insert(0, 'src')

from Training.Data_Wrangling.mosi_audio_extractor import (
    parse_segment_id,
    get_segment_timestamps,
    extract_audio_segment
)
from Training.Data_Wrangling.mosi_frame_extractor import extract_frame

# Configuration
MOSI_DATA_PATH = 'data/cmumosi/mosi/'
VIDEO_DIR = 'data/cmumosi/videos'
AUDIO_DIR = 'data/cmumosi/audio'
FRAMES_DIR = 'data/cmumosi/frames'
DOWNLOAD_MANIFEST = 'data/cmumosi/videos/download_manifest.json'


def get_downloaded_videos():
    """Get list of successfully downloaded video IDs from manifest."""
    if not os.path.exists(DOWNLOAD_MANIFEST):
        print(f"[ERROR] Download manifest not found: {DOWNLOAD_MANIFEST}")
        print("Run: python scripts/data_wrangling/download_all_mosi_videos.py")
        return []

    with open(DOWNLOAD_MANIFEST, 'r') as f:
        manifest = json.load(f)

    downloaded = []
    for video_id, info in manifest['videos'].items():
        if info['status'] == 'already_exists':
            downloaded.append(video_id)

    print(f"[OK] Found {len(downloaded)} downloaded videos")
    return sorted(downloaded)


def get_all_segments_from_preprocessed(downloaded_videos: set) -> List[str]:
    """Get fine-grained opinion segment IDs from preprocessed pkl files."""
    all_segments = []

    print("Loading segment IDs from preprocessed pkl files...")

    for split in ['train', 'valid', 'test']:
        pkl_path = os.path.join(MOSI_DATA_PATH, f'preprocessed_{split}.pkl')

        if not os.path.exists(pkl_path):
            print(f"[WARNING] {pkl_path} not found, skipping {split} split")
            continue

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            segment_ids = data['segment_ids']  # List of "video_id[num]" strings

        # Filter to downloaded videos only
        for seg_id in segment_ids:
            video_id = seg_id.split('[')[0]
            if video_id in downloaded_videos:
                all_segments.append((seg_id, split))  # Track split for stats

    # Show breakdown by split
    print(f"\n[OK] Found {len(all_segments)} opinion segments from {len(downloaded_videos)} videos")

    split_counts = {'train': 0, 'valid': 0, 'test': 0}
    for _, split in all_segments:
        split_counts[split] += 1

    print(f"  Train: {split_counts['train']} segments")
    print(f"  Valid: {split_counts['valid']} segments")
    print(f"  Test:  {split_counts['test']} segments")

    # Show first few segment IDs to understand format
    print(f"\nFirst few segment IDs (to understand format):")
    for seg_id, split in sorted(all_segments)[:5]:
        print(f"  {seg_id} ({split})")

    # Return just segment IDs (drop split info)
    return sorted([seg_id for seg_id, _ in all_segments])


def main():
    print("=" * 80)
    print("Extracting Audio and Frames for ALL Downloaded MOSI Videos")
    print("=" * 80)

    # Create output directories
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Get downloaded videos
    print("\n[Step 1/3] Getting downloaded videos...")
    downloaded_videos = set(get_downloaded_videos())

    if not downloaded_videos:
        print("[ERROR] No downloaded videos found!")
        print("  Run: python scripts/data_wrangling/download_all_mosi_videos.py")
        return 1

    # Get fine-grained segment IDs from preprocessed data
    print("\n[Step 2/3] Loading fine-grained opinion segments...")
    all_segments = get_all_segments_from_preprocessed(downloaded_videos)

    if not all_segments:
        print("[ERROR] No segments found. Check that:")
        print("  1. Videos are downloaded (download_manifest.json exists)")
        print("  2. Preprocessed pkl files exist (data/cmumosi/mosi/preprocessed_*.pkl)")
        return 1

    # Count segments per video
    video_counts = {}
    for seg_id in all_segments:
        video_id = seg_id.split('[')[0] if '[' in seg_id else seg_id[:11]
        video_counts[video_id] = video_counts.get(video_id, 0) + 1

    print(f"\nSegment breakdown by video (showing first 10):")
    for i, (video_id, count) in enumerate(sorted(video_counts.items(), key=lambda x: -x[1])[:10]):
        print(f"  {video_id}: {count} segments")
    if len(video_counts) > 10:
        print(f"  ... and {len(video_counts) - 10} more videos")

    # Extract audio
    print(f"\n[Step 2/3] Extracting audio for {len(all_segments)} segments...")
    print("-" * 80)

    audio_success = 0
    audio_failed = 0
    audio_skipped = 0

    for seg_id in all_segments:
        # Parse segment ID (handle different formats)
        if '[' in seg_id:
            video_id, segment_num = parse_segment_id(seg_id)
        else:
            # Segment ID without brackets - use as-is
            video_id = seg_id[:11] if len(seg_id) >= 11 else seg_id
            segment_num = 0  # Default segment number

        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        audio_path = os.path.join(AUDIO_DIR, f"{seg_id.replace('[', '_').replace(']', '')}.wav")

        # Skip if already exists (idempotent)
        if os.path.exists(audio_path):
            audio_skipped += 1
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

    print(f"\nAudio extraction: {audio_success} new, {audio_skipped} skipped, {audio_failed} failed ({len(all_segments)} total)")

    # Extract frames
    print(f"\n[Step 3/3] Extracting frames for {len(all_segments)} segments...")
    print("-" * 80)

    frame_success = 0
    frame_failed = 0
    frame_skipped = 0

    for seg_id in all_segments:
        # Parse segment ID (handle different formats)
        if '[' in seg_id:
            video_id, segment_num = parse_segment_id(seg_id)
        else:
            # Segment ID without brackets - use as-is
            video_id = seg_id[:11] if len(seg_id) >= 11 else seg_id
            segment_num = 0  # Default segment number

        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        frame_path = os.path.join(FRAMES_DIR, f"{seg_id.replace('[', '_').replace(']', '')}.jpg")

        # Skip if already exists (idempotent)
        if os.path.exists(frame_path):
            frame_skipped += 1
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

    print(f"\nFrame extraction: {frame_success} new, {frame_skipped} skipped, {frame_failed} failed ({len(all_segments)} total)")

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total segments processed: {len(all_segments)}")
    print(f"Audio files: {audio_success + audio_skipped}/{len(all_segments)} ({audio_success} new, {audio_skipped} existing)")
    print(f"Frame files: {frame_success + frame_skipped}/{len(all_segments)} ({frame_success} new, {frame_skipped} existing)")
    print()

    # Output sizes
    total_audio = audio_success + audio_skipped
    total_frames = frame_success + frame_skipped

    if total_audio > 0 and total_frames > 0:
        print("[OK] Extraction successful! Multimodal training data ready.")
        print(f"\nOutput locations:")
        print(f"  Audio:  {AUDIO_DIR}  ({total_audio} files)")
        print(f"  Frames: {FRAMES_DIR}  ({total_frames} files)")
        print("\nNext steps:")
        print("  1. Verify counts match:")
        print(f"     ls {AUDIO_DIR} | wc -l   # Should be ~{total_audio}")
        print(f"     ls {FRAMES_DIR} | wc -l   # Should be ~{total_frames}")
        print("  2. Test with training:")
        print("     python src/Training/train_raw_encoders.py --max-samples 10")
    else:
        print("[WARNING] Extraction incomplete. Check errors above.")

    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
