"""
Simple test using ONLY the 5 extracted test segments.
"""

import sys
import os

sys.path.insert(0, 'src')

from mmsdk import mmdatasdk
from Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from PIL import Image
import librosa


# Test video IDs we extracted
TEST_VIDEO_IDS = [
    'BvYR0L6f2Ig',
    'Vj1wYRQjB-o',
    '9J25DZhivz8',
    'tmZoasNr4rU',
    'Bfr499ggo-0',
]


def main():
    print("=" * 80)
    print("Simple Dataset Test - Using Only 5 Extracted Segments")
    print("=" * 80)

    # Load raw MOSI to verify our 5 segments exist
    print("\n[Step 1/6] Loading raw MOSI data...")
    print("-" * 80)

    raw_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.raw, 'data/cmumosi/mosi/')
    highlevel_data = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, 'data/cmumosi/mosi/')
    highlevel_data.add_computational_sequences(mmdatasdk.cmu_mosi.labels, 'data/cmumosi/mosi/')

    words_seq = raw_data.computational_sequences['words']
    labels_seq = highlevel_data.computational_sequences['Opinion Segment Labels']

    # Filter to our test videos
    test_segments = []
    for seg_id in words_seq.data.keys():
        # Segment IDs are just the video ID (11 characters)
        video_id = seg_id[:11] if len(seg_id) >= 11 else seg_id
        if video_id in TEST_VIDEO_IDS:
            test_segments.append(seg_id)

    print(f"\n[OK] Found {len(test_segments)} segments from our test videos")
    for seg_id in sorted(test_segments):
        print(f"  - {seg_id}")

    # Manually test loading each segment
    print("\n[Step 2/6] Testing manual data loading...")
    print("-" * 80)

    for seg_id in sorted(test_segments)[:1]:  # Test first one
        # Get text
        words_data = words_seq.data[seg_id]
        if isinstance(words_data['features'], (list,)):
            import numpy as np
            if isinstance(words_data['features'], np.ndarray):
                text = ' '.join([str(w) for w in words_data['features']])
            else:
                text = ' '.join([str(w) for w in words_data['features']])
        else:
            text = str(words_data['features'])

        # Get label
        label = float(labels_seq.data[seg_id]['features'][0])

        # Get file paths
        file_basename = seg_id.replace('[', '_').replace(']', '')
        audio_path = f"data/cmumosi/audio/{file_basename}.wav"
        video_path = f"data/cmumosi/frames/{file_basename}.jpg"

        print(f"\nSegment: {seg_id}")
        print(f"  Text: '{text[:80]}...'")
        print(f"  Label: {label}")
        print(f"  Audio: {audio_path} - Exists: {os.path.exists(audio_path)}")
        print(f"  Video: {video_path} - Exists: {os.path.exists(video_path)}")

        if not os.path.exists(audio_path):
            print(f"\n[ERROR] Audio file not found!")
            print(f"  Looking for: {audio_path}")
            # List what's actually in the audio directory
            audio_dir_files = os.listdir('data/cmumosi/audio')
            print(f"  Files in audio dir: {audio_dir_files[:5]}")
            return

        # Test encoders
        print("\n[Step 3/6] Testing Text_to_Vec...")
        text_encoder = Text_to_Vec()
        text_vec = text_encoder(text)
        print(f"[OK] Text encoding: {text_vec.shape}")

        print("\n[Step 4/6] Testing Audio_to_Vec...")
        audio_encoder = Audio_to_Vec()
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"  Loaded audio: {len(waveform)/sr:.2f}s")
        audio_vec = audio_encoder(waveform)
        print(f"[OK] Audio encoding: {audio_vec.shape}")

        print("\n[Step 5/6] Testing Image_to_Vec...")
        image_encoder = Image_to_Vec()
        frame = Image.open(video_path).convert('RGB')
        print(f"  Loaded frame: {frame.size}")
        image_vec = image_encoder(frame)
        print(f"[OK] Image encoding: {image_vec.shape}")

        break  # Just test first segment

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test segments available: {len(test_segments)}")
    print(f"All encoders: [OK]")
    print()
    print("Next steps:")
    print("  1. Fix dataset loader to work with these segments")
    print("  2. Or extract ALL MOSI segments (not just test videos)")
    print("=" * 80)


if __name__ == "__main__":
    main()
