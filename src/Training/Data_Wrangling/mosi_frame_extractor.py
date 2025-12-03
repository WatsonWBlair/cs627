"""
MOSI Frame Extractor

Extracts representative frames from downloaded MOSI videos using timestamps.
Saves as JPG images for ViT compatibility.

Usage:
    from mosi_frame_extractor import extract_frames
    manifest = extract_frames(
        video_dir='data/cmumosi/videos',
        output_dir='data/cmumosi/frames',
        split='train'
    )
"""

import os
import pickle
import json
from typing import List, Dict, Tuple
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np


def parse_segment_id(seg_id: str) -> Tuple[str, int]:
    """
    Parse MOSI segment ID into video ID and segment number.

    Args:
        seg_id: MOSI segment ID (e.g., 'BvYR0L6f2Ig[2]')

    Returns:
        Tuple of (video_id, segment_num)
    """
    parts = seg_id.split('[')
    video_id = parts[0]
    segment_num = int(parts[1].rstrip(']'))
    return video_id, segment_num


def extract_frame(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    strategy: str = 'middle'
) -> Dict:
    """
    Extract a representative frame from video segment.

    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save frame (JPG)
        strategy: Frame selection strategy ('middle', 'first', 'last')

    Returns:
        Dictionary with extraction results:
            {'status': 'success'/'failed', 'path': str, 'timestamp': float, 'error': str}
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {
                'status': 'failed',
                'error': f'Could not open video: {video_path}'
            }

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Select frame based on strategy
        if strategy == 'middle':
            target_frame = (start_frame + end_frame) // 2
        elif strategy == 'first':
            target_frame = start_frame
        elif strategy == 'last':
            target_frame = end_frame
        else:
            target_frame = (start_frame + end_frame) // 2

        # Ensure frame is within bounds
        target_frame = max(0, min(target_frame, total_frames - 1))

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        # Read frame
        ret, frame = cap.read()

        if not ret or frame is None:
            cap.release()
            return {
                'status': 'failed',
                'error': f'Could not read frame {target_frame} from video'
            }

        # Convert BGR to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image and save as JPG
        pil_image = Image.fromarray(frame_rgb)
        pil_image.save(output_path, 'JPEG', quality=95)

        cap.release()

        # Get actual timestamp
        timestamp = target_frame / fps

        return {
            'status': 'success',
            'path': output_path,
            'timestamp': timestamp,
            'frame_number': target_frame,
            'resolution': f"{frame.shape[1]}x{frame.shape[0]}"
        }

    except FileNotFoundError:
        return {
            'status': 'video_not_found',
            'error': f'Video file not found: {video_path}'
        }

    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)[:200]
        }


def get_segment_timestamps(seg_id: str, mosi_data_path: str) -> Tuple[float, float]:
    """
    Get start and end timestamps for a segment from MOSI data.

    Args:
        seg_id: MOSI segment ID
        mosi_data_path: Path to MOSI dataset

    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    try:
        from mmsdk import mmdatasdk
    except ImportError:
        raise ImportError(
            "CMU-MultimodalSDK not installed. Install it:\n"
            "git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git\n"
            "cd CMU-MultimodalSDK && pip install ."
        )

    # Load MOSI data to get timestamps
    try:
        # Load words sequence (has timestamps)
        raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, mosi_data_path)
        words_seq = raw_data.computational_sequences['words']

        # Get segment data
        if seg_id in words_seq.data:
            segment_data = words_seq.data[seg_id]
            intervals = segment_data['intervals']

            # intervals is array of [start, end] timestamps
            start_time = float(intervals[0][0])
            end_time = float(intervals[-1][1])

            return start_time, end_time
        else:
            raise ValueError(f"Segment {seg_id} not found in MOSI data")

    except Exception as e:
        raise RuntimeError(f"Failed to get timestamps for {seg_id}: {e}")


def extract_frames(
    video_dir: str = 'data/cmumosi/videos',
    output_dir: str = 'data/cmumosi/frames',
    mosi_data_path: str = 'data/cmumosi/mosi/',
    split: str = 'train',
    strategy: str = 'middle'
) -> Dict:
    """
    Extract frames for all videos in a split.

    Args:
        video_dir: Directory containing downloaded videos
        output_dir: Directory to save frames
        mosi_data_path: Path to MOSI dataset
        split: Dataset split ('train', 'valid', or 'test')
        strategy: Frame selection strategy ('middle', 'first', 'last')

    Returns:
        Manifest dictionary with extraction results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load segment IDs from preprocessed split
    preprocessed_path = os.path.join(mosi_data_path, f'preprocessed_{split}.pkl')

    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {preprocessed_path}"
        )

    print(f"\nLoading {split} split from: {preprocessed_path}")
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
        segment_ids = data['segment_ids']

    print(f"Found {len(segment_ids)} segments in {split} split")

    # Create manifest
    manifest = {
        'split': split,
        'total_segments': len(segment_ids),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'strategy': strategy,
        'segments': {}
    }

    print(f"\nExtracting frames (strategy: {strategy})...")
    print("=" * 80)

    for seg_id in tqdm(segment_ids, desc="Extracting frames"):
        video_id, segment_num = parse_segment_id(seg_id)

        # Path to video file
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        # Output path for frame
        frame_filename = f"{video_id}_seg{segment_num}.jpg"
        output_path = os.path.join(output_dir, frame_filename)

        # Skip if already extracted
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 1000:  # Valid file
                manifest['segments'][seg_id] = {
                    'status': 'already_exists',
                    'path': output_path
                }
                manifest['successful_extractions'] += 1
                continue

        # Check if video exists
        if not os.path.exists(video_path):
            manifest['segments'][seg_id] = {
                'status': 'video_not_found',
                'video_path': video_path
            }
            manifest['failed_extractions'] += 1
            continue

        # Get timestamps
        try:
            start_time, end_time = get_segment_timestamps(seg_id, mosi_data_path)
        except Exception as e:
            manifest['segments'][seg_id] = {
                'status': 'timestamp_error',
                'error': str(e)[:200]
            }
            manifest['failed_extractions'] += 1
            continue

        # Extract frame
        result = extract_frame(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            output_path=output_path,
            strategy=strategy
        )

        manifest['segments'][seg_id] = result

        if result['status'] == 'success':
            manifest['successful_extractions'] += 1
        else:
            manifest['failed_extractions'] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total segments: {manifest['total_segments']}")
    print(f"Successful extractions: {manifest['successful_extractions']}")
    print(f"Failed extractions: {manifest['failed_extractions']}")

    success_rate = manifest['successful_extractions'] / manifest['total_segments'] if manifest['total_segments'] > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1%}")

    # Failure breakdown
    failure_types = {}
    for seg_id, result in manifest['segments'].items():
        if result['status'] != 'success' and result['status'] != 'already_exists':
            failure_types[result['status']] = failure_types.get(result['status'], 0) + 1

    if failure_types:
        print("\nFailure Breakdown:")
        for status, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {status}: {count}")

    # Save manifest
    manifest_path = os.path.join(output_dir, f'extraction_manifest_{split}.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")

    print("=" * 80)

    return manifest


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract frames from MOSI videos')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--video-dir', type=str, default='data/cmumosi/videos',
                       help='Directory containing videos')
    parser.add_argument('--output-dir', type=str, default='data/cmumosi/frames',
                       help='Output directory for frames')
    parser.add_argument('--strategy', type=str, default='middle',
                       choices=['middle', 'first', 'last'],
                       help='Frame selection strategy')

    args = parser.parse_args()

    print("=" * 80)
    print(f"MOSI Frame Extractor - {args.split.upper()} Split")
    print("=" * 80)

    manifest = extract_frames(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        split=args.split,
        strategy=args.strategy
    )

    print("\nExtraction complete!")
    print(f"Frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
