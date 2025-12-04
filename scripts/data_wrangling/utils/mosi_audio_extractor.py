"""
MOSI Audio Extractor

Extracts audio segments from downloaded MOSI videos using timestamps.
Resamples to 16kHz mono for Whisper compatibility.

Usage:
    from mosi_audio_extractor import extract_audio_segments
    manifest = extract_audio_segments(
        video_dir='data/cmumosi/videos',
        output_dir='data/cmumosi/audio',
        split='train'
    )
"""

import os
import pickle
import json
from typing import List, Dict, Tuple
from pathlib import Path
import librosa
import soundfile as sf
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


def extract_audio_segment(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    target_sr: int = 16000
) -> Dict:
    """
    Extract audio segment from video file.

    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save audio segment
        target_sr: Target sampling rate (default: 16000 for Whisper)

    Returns:
        Dictionary with extraction results:
            {'status': 'success'/'failed', 'path': str, 'duration': float, 'error': str}
    """
    try:
        # Load audio from video
        # librosa can handle video files and extract audio
        audio, sr = librosa.load(video_path, sr=None, mono=True)

        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Extract segment
        audio_segment = audio[start_sample:end_sample]

        # Resample to target rate if needed
        if sr != target_sr:
            audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=target_sr)

        # Save as WAV
        sf.write(output_path, audio_segment, target_sr)

        duration = len(audio_segment) / target_sr

        return {
            'status': 'success',
            'path': output_path,
            'duration': duration,
            'sampling_rate': target_sr,
            'samples': len(audio_segment)
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
        seg_id: MOSI segment ID (format: video_id[segment_num], e.g., "03bSnISJMiM[0]")
        mosi_data_path: Path to MOSI dataset

    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    try:
        from mmsdk import mmdatasdk
        import numpy as np
    except ImportError:
        raise ImportError(
            "CMU-MultimodalSDK not installed. Install it:\n"
            "git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git\n"
            "cd CMU-MultimodalSDK && pip install ."
        )

    # Load MOSI data to get timestamps
    try:
        # Parse segment ID to get video_id and segment number
        if '[' not in seg_id:
            raise ValueError(f"Segment ID must be in format 'video_id[num]', got: {seg_id}")

        video_id, segment_num = parse_segment_id(seg_id)

        # Load labels (opinion segments with timestamps)
        highlevel_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, mosi_data_path)
        highlevel_data.add_computational_sequences(mmdatasdk.cmu_mosi.labels, mosi_data_path)
        labels_seq = highlevel_data.computational_sequences['Opinion Segment Labels']

        # Get video's label data
        if video_id not in labels_seq.data:
            raise ValueError(f"Video {video_id} not found in MOSI labels")

        labels_data = labels_seq.data[video_id]
        intervals = np.array(labels_data['intervals'])  # Shape: (num_segments, 2)

        # Check if segment number is valid
        if segment_num >= len(intervals):
            raise ValueError(
                f"Segment number {segment_num} out of range for video {video_id} "
                f"(has {len(intervals)} segments)"
            )

        # Extract timestamps for this specific segment
        start_time = float(intervals[segment_num][0])
        end_time = float(intervals[segment_num][1])

        return start_time, end_time

    except Exception as e:
        raise RuntimeError(f"Failed to get timestamps for {seg_id}: {e}")


def extract_audio_segments(
    video_dir: str = 'data/cmumosi/videos',
    output_dir: str = 'data/cmumosi/audio',
    mosi_data_path: str = 'data/cmumosi/mosi/',
    split: str = 'train',
    target_sr: int = 16000
) -> Dict:
    """
    Extract audio segments for all videos in a split.

    Args:
        video_dir: Directory containing downloaded videos
        output_dir: Directory to save audio segments
        mosi_data_path: Path to MOSI dataset
        split: Dataset split ('train', 'valid', or 'test')
        target_sr: Target sampling rate for audio

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
        'segments': {}
    }

    print(f"\nExtracting audio segments...")
    print("=" * 80)

    for seg_id in tqdm(segment_ids, desc="Extracting audio"):
        video_id, segment_num = parse_segment_id(seg_id)

        # Path to video file
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        # Output path for audio
        audio_filename = f"{video_id}_seg{segment_num}.wav"
        output_path = os.path.join(output_dir, audio_filename)

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

        # Extract audio
        result = extract_audio_segment(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            output_path=output_path,
            target_sr=target_sr
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

    parser = argparse.ArgumentParser(description='Extract audio segments from MOSI videos')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--video-dir', type=str, default='data/cmumosi/videos',
                       help='Directory containing videos')
    parser.add_argument('--output-dir', type=str, default='data/cmumosi/audio',
                       help='Output directory for audio')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Target sampling rate (default: 16000 for Whisper)')

    args = parser.parse_args()

    print("=" * 80)
    print(f"MOSI Audio Extractor - {args.split.upper()} Split")
    print("=" * 80)

    manifest = extract_audio_segments(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        split=args.split,
        target_sr=args.sample_rate
    )

    print("\nExtraction complete!")
    print(f"Audio files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
