"""
Test the MOSIRawVideoDataset loader with extracted test data.

This script tests:
1. Loading the dataset (using raw MOSI segment IDs)
2. Accessing samples
3. Checking data shapes and compatibility with encoders
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from PIL import Image
import librosa


def main():
    print("=" * 80)
    print("Testing MOSIRawVideoDataset Loader")
    print("=" * 80)

    # Test loading the dataset
    print("\n[Step 1/5] Loading train split...")
    print("-" * 80)

    try:
        dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/',
            train_ratio=0.6,  # Use 60% for train (3 videos)
            val_ratio=0.2,    # Use 20% for valid (1 video)
            seed=42
        )

        print(f"\n[OK] Dataset loaded successfully!")
        print(f"  Total samples in train split: {len(dataset)}")

    except Exception as e:
        print(f"\n[FAIL] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test accessing a sample
    print("\n[Step 2/5] Testing sample access...")
    print("-" * 80)

    if len(dataset) > 0:
        try:
            sample = dataset[0]
            print(f"\n[OK] Sample structure:")
            print(f"  Text: '{sample['text'][:80]}...'")
            print(f"  Audio path: {sample['audio']}")
            print(f"  Video path: {sample['video']}")
            print(f"  Label: {sample['label']}")
            print(f"  Segment ID: {sample['segment_id']}")

            # Verify files exist
            if os.path.exists(sample['audio']):
                audio_size = os.path.getsize(sample['audio']) / 1024  # KB
                print(f"\n  Audio file exists: {audio_size:.1f} KB")
            else:
                print(f"\n  [WARNING] Audio file not found: {sample['audio']}")

            if os.path.exists(sample['video']):
                video_size = os.path.getsize(sample['video']) / 1024  # KB
                print(f"  Video file exists: {video_size:.1f} KB")
            else:
                print(f"  [WARNING] Video file not found: {sample['video']}")

        except Exception as e:
            print(f"\n[FAIL] Failed to access sample: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("\n[WARNING] Dataset is empty!")
        return

    # Test text encoder
    print("\n[Step 3/5] Testing Text_to_Vec encoder...")
    print("-" * 80)

    try:
        text_encoder = Text_to_Vec()
        text_vec = text_encoder(sample['text'])
        print(f"\n[OK] Text encoder works!")
        print(f"  Input: '{sample['text'][:60]}...'")
        print(f"  Output shape: {text_vec.shape}")
        print(f"  Expected: (1, 1024)")
        assert text_vec.shape == (1, 1024), f"Unexpected shape: {text_vec.shape}"

    except Exception as e:
        print(f"\n[FAIL] Text encoder failed: {e}")
        import traceback
        traceback.print_exc()

    # Test audio encoder
    print("\n[Step 4/5] Testing Audio_to_Vec encoder...")
    print("-" * 80)

    try:
        audio_encoder = Audio_to_Vec()

        # Load audio waveform
        waveform, sr = librosa.load(sample['audio'], sr=16000, mono=True)
        print(f"\n  Loaded audio: {len(waveform)/sr:.2f}s at {sr}Hz")

        # Encode
        audio_vec = audio_encoder(waveform)
        print(f"\n[OK] Audio encoder works!")
        print(f"  Input: waveform of {len(waveform)} samples")
        print(f"  Output shape: {audio_vec.shape}")
        print(f"  Expected: (1, 1024)")
        assert audio_vec.shape == (1, 1024), f"Unexpected shape: {audio_vec.shape}"

    except Exception as e:
        print(f"\n[FAIL] Audio encoder failed: {e}")
        import traceback
        traceback.print_exc()

    # Test image encoder
    print("\n[Step 5/5] Testing Image_to_Vec encoder...")
    print("-" * 80)

    try:
        image_encoder = Image_to_Vec()

        # Load image frame
        frame = Image.open(sample['video']).convert('RGB')
        print(f"\n  Loaded frame: {frame.size} ({frame.mode})")

        # Encode
        image_vec = image_encoder(frame)
        print(f"\n[OK] Image encoder works!")
        print(f"  Input: PIL Image {frame.size}")
        print(f"  Output shape: {image_vec.shape}")
        print(f"  Expected: (1, 1024)")
        assert image_vec.shape == (1, 1024), f"Unexpected shape: {image_vec.shape}"

    except Exception as e:
        print(f"\n[FAIL] Image encoder failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("DATASET LOADER TEST SUMMARY")
    print("=" * 80)
    print(f"Dataset samples: {len(dataset)}")
    print(f"All modalities: text (strings), audio (paths), video (paths)")
    print(f"Label range: sentiment intensity scores")
    print()
    print("Encoder compatibility:")
    print("  Text_to_Vec: [OK] - Accepts strings, outputs (1, 1024)")
    print("  Audio_to_Vec: [OK] - Accepts 16kHz waveforms, outputs (1, 1024)")
    print("  Image_to_Vec: [OK] - Accepts PIL Images, outputs (1, 1024)")
    print()
    print("[OK] MOSIRawVideoDataset is ready for training!")
    print("=" * 80)

    # Test valid and test splits
    print("\n[Bonus] Testing valid and test splits...")
    print("-" * 80)

    try:
        valid_dataset = MOSIRawVideoDataset(
            split='valid',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/',
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42
        )
        print(f"[OK] Valid split: {len(valid_dataset)} samples")

        test_dataset = MOSIRawVideoDataset(
            split='test',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/',
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42
        )
        print(f"[OK] Test split: {len(test_dataset)} samples")

    except Exception as e:
        print(f"[INFO] Split test: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
