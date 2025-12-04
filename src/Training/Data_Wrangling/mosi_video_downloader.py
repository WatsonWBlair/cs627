"""
MOSI Video Downloader

Downloads YouTube videos from CMU-MOSI dataset using segment IDs.

Usage:
    from mosi_video_downloader import download_mosi_videos
    manifest = download_mosi_videos(segment_ids, output_dir='data/cmumosi/videos')

Expected success rate: 55-60% (based on test_mosi_video_access.py results)
"""

import os
import pickle
import json
from typing import List, Dict
from pathlib import Path
import yt_dlp
from tqdm import tqdm
import concurrent.futures


def parse_segment_id(seg_id: str) -> str:
    """
    Extract YouTube video ID from MOSI segment ID.

    Args:
        seg_id: MOSI segment ID (e.g., 'BvYR0L6f2Ig[2]')

    Returns:
        YouTube video ID (e.g., 'BvYR0L6f2Ig')
    """
    return seg_id.split('[')[0]


def download_single_video(video_id: str, output_dir: str) -> Dict:
    """
    Download a single YouTube video.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save video

    Returns:
        Dictionary with download results:
            {'video_id': str, 'status': 'success'/'failed', 'path': str, 'error': str}
    """
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    # Skip if already downloaded
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # More than 1KB = valid file
            return {
                'video_id': video_id,
                'status': 'already_exists',
                'path': output_path,
                'size_mb': file_size / (1024 * 1024)
            }

    # Configure yt-dlp
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,  # Continue even if download fails
        'no_color': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)

            if info and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                return {
                    'video_id': video_id,
                    'status': 'success',
                    'path': output_path,
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'size_mb': file_size / (1024 * 1024)
                }
            else:
                return {
                    'video_id': video_id,
                    'status': 'failed',
                    'error': 'Download completed but file not found'
                }

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if 'Private video' in error_msg:
            status = 'private'
        elif 'Video unavailable' in error_msg:
            status = 'unavailable'
        elif 'This video has been removed' in error_msg:
            status = 'removed'
        else:
            status = 'download_error'

        return {
            'video_id': video_id,
            'status': status,
            'error': error_msg[:200]
        }

    except (OSError, ConnectionError, TimeoutError) as e:
        # File system or network errors
        return {
            'video_id': video_id,
            'status': 'network_error',
            'error': str(e)[:200]
        }

    except (KeyError, AttributeError, RuntimeError) as e:
        # Parsing or runtime errors
        return {
            'video_id': video_id,
            'status': 'error',
            'error': str(e)[:200]
        }


def download_mosi_videos(
    segment_ids: List[str],
    output_dir: str = 'data/cmumosi/videos',
    max_workers: int = 4,
    save_manifest: bool = True
) -> Dict:
    """
    Download YouTube videos for MOSI segments.

    Args:
        segment_ids: List of MOSI segment IDs
        output_dir: Directory to save videos
        max_workers: Number of parallel download threads
        save_manifest: Whether to save manifest JSON file

    Returns:
        Manifest dictionary:
            {
                'total_segments': int,
                'unique_videos': int,
                'successful_downloads': int,
                'already_exists': int,
                'failed_downloads': int,
                'videos': {video_id: result_dict, ...}
            }
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique video IDs
    print(f"\nExtracting unique video IDs from {len(segment_ids)} segments...")
    video_ids = [parse_segment_id(sid) for sid in segment_ids]
    unique_video_ids = list(set(video_ids))
    print(f"Found {len(unique_video_ids)} unique videos")

    # Download videos
    print(f"\nDownloading videos with {max_workers} parallel workers...")
    print("This may take several hours for the full dataset.")
    print("=" * 80)

    manifest = {
        'total_segments': len(segment_ids),
        'unique_videos': len(unique_video_ids),
        'successful_downloads': 0,
        'already_exists': 0,
        'failed_downloads': 0,
        'videos': {}
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_video = {
            executor.submit(download_single_video, vid_id, output_dir): vid_id
            for vid_id in unique_video_ids
        }

        # Process completed downloads
        with tqdm(total=len(unique_video_ids), desc="Downloading") as pbar:
            for future in concurrent.futures.as_completed(future_to_video):
                video_id = future_to_video[future]

                try:
                    result = future.result()
                    manifest['videos'][video_id] = result

                    # Update counters
                    if result['status'] == 'success':
                        manifest['successful_downloads'] += 1
                        pbar.set_postfix({
                            'success': manifest['successful_downloads'],
                            'failed': manifest['failed_downloads']
                        })
                    elif result['status'] == 'already_exists':
                        manifest['already_exists'] += 1
                    else:
                        manifest['failed_downloads'] += 1

                except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError) as e:
                    # Future execution errors
                    manifest['videos'][video_id] = {
                        'video_id': video_id,
                        'status': 'timeout',
                        'error': str(e)[:200]
                    }
                    manifest['failed_downloads'] += 1

                except (KeyError, AttributeError) as e:
                    # Result processing errors
                    manifest['videos'][video_id] = {
                        'video_id': video_id,
                        'status': 'exception',
                        'error': str(e)[:200]
                    }
                    manifest['failed_downloads'] += 1

                pbar.update(1)

    # Print summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Total segments: {manifest['total_segments']}")
    print(f"Unique videos: {manifest['unique_videos']}")
    print(f"Successful downloads: {manifest['successful_downloads']}")
    print(f"Already existed: {manifest['already_exists']}")
    print(f"Failed downloads: {manifest['failed_downloads']}")

    total_available = manifest['successful_downloads'] + manifest['already_exists']
    success_rate = total_available / manifest['unique_videos'] if manifest['unique_videos'] > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1%}")

    # Failure breakdown
    failure_types = {}
    for video_id, result in manifest['videos'].items():
        if result['status'] not in ['success', 'already_exists']:
            failure_types[result['status']] = failure_types.get(result['status'], 0) + 1

    if failure_types:
        print("\nFailure Breakdown:")
        for status, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {status}: {count}")

    # Save manifest
    if save_manifest:
        manifest_path = os.path.join(output_dir, 'download_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved to: {manifest_path}")

    print("=" * 80)

    return manifest


def download_from_split(
    split: str = 'train',
    data_path: str = 'data/cmumosi/mosi/',
    output_dir: str = 'data/cmumosi/videos',
    max_workers: int = 4
) -> Dict:
    """
    Download videos for a specific MOSI split (train/valid/test).

    Args:
        split: Dataset split ('train', 'valid', or 'test')
        data_path: Path to MOSI preprocessed data
        output_dir: Directory to save videos
        max_workers: Number of parallel download threads

    Returns:
        Download manifest dictionary
    """
    # Load segment IDs from preprocessed split
    preprocessed_path = os.path.join(data_path, f'preprocessed_{split}.pkl')

    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {preprocessed_path}\n"
            f"Please ensure MOSI dataset is preprocessed."
        )

    print(f"Loading {split} split from: {preprocessed_path}")
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
        segment_ids = data['segment_ids']

    print(f"Found {len(segment_ids)} segments in {split} split")

    # Download videos
    manifest = download_mosi_videos(
        segment_ids=segment_ids,
        output_dir=output_dir,
        max_workers=max_workers,
        save_manifest=True
    )

    # Save split-specific manifest
    split_manifest_path = os.path.join(output_dir, f'manifest_{split}.json')
    with open(split_manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Download MOSI YouTube videos')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to download')
    parser.add_argument('--output-dir', type=str, default='data/cmumosi/videos',
                       help='Output directory for videos')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel download workers')

    args = parser.parse_args()

    print("=" * 80)
    print(f"MOSI Video Downloader - {args.split.upper()} Split")
    print("=" * 80)

    manifest = download_from_split(
        split=args.split,
        output_dir=args.output_dir,
        max_workers=args.workers
    )

    print("\nDownload complete!")
    print(f"Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
