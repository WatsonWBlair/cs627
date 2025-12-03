"""
Test the training dataloader with our 5 extracted test videos.

Uses 100% of the 5 extracted videos for the test.
"""

import sys
import os
sys.path.insert(0, 'src')

from Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
from torch.utils.data import DataLoader
from typing import List, Dict
import librosa
from PIL import Image as PILImage
import numpy as np


def collate_fn_raw_video(batch: List[Dict]) -> Dict:
    """Lazy loading collate function."""
    texts = [sample['text'] for sample in batch]

    # Lazy load audio waveforms
    audio_waveforms = []
    for sample in batch:
        try:
            waveform, sr = librosa.load(sample['audio'], sr=16000, mono=True)
            audio_waveforms.append(waveform)
        except Exception as e:
            print(f"[WARNING] Failed to load audio: {e}")
            audio_waveforms.append(np.zeros(16000, dtype=np.float32))

    # Lazy load video frames
    video_frames = []
    for sample in batch:
        try:
            frame = PILImage.open(sample['video']).convert('RGB')
            video_frames.append(frame)
        except Exception as e:
            print(f"[WARNING] Failed to load frame: {e}")
            video_frames.append(PILImage.new('RGB', (224, 224), color='black'))

    return {
        'text': texts,
        'audio': audio_waveforms,
        'video': video_frames
    }


def main():
    print("=" * 80)
    print("Testing Dataloader with 5 Extracted Test Videos")
    print("=" * 80)

    # Load dataset with 100% train split (uses all 5 videos)
    print("\n[Step 1/3] Loading dataset (100% train split)...")
    print("-" * 80)

    try:
        dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/',
            train_ratio=1.0,  # Use 100% for train
            val_ratio=0.0,
            seed=42
        )
        print(f"[OK] Loaded {len(dataset)} training samples")

        if len(dataset) == 0:
            print("[ERROR] Dataset is empty!")
            print("This means none of the videos in the train split have extracted audio/frames.")
            print("We only extracted 5 videos total, so we need to use all of them.")
            return

    except Exception as e:
        print(f"[FAIL] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create dataloader
    print("\n[Step 2/3] Creating dataloader and loading batch...")
    print("-" * 80)

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=min(2, len(dataset)),  # Use batch size of 2 or fewer
            shuffle=False,
            collate_fn=collate_fn_raw_video,
            num_workers=0
        )

        # Load first batch
        batch = next(iter(dataloader))
        print(f"[OK] Loaded batch with {len(batch['text'])} samples")
        print(f"\nBatch details:")
        for i in range(len(batch['text'])):
            print(f"\n  Sample {i}:")
            print(f"    Text: '{batch['text'][i][:60]}...'")
            print(f"    Audio: {batch['audio'][i].shape} ({len(batch['audio'][i])/16000:.2f}s)")
            print(f"    Video: {batch['video'][i].size} {batch['video'][i].mode}")

    except Exception as e:
        print(f"[FAIL] Failed to create dataloader or load batch: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with encoders
    print("\n[Step 3/3] Testing with encoders...")
    print("-" * 80)

    try:
        from Encoders.text_2_vec import Text_to_Vec
        from Encoders.wav_2_vec import Audio_to_Vec
        from Encoders.img_2_vec import Image_to_Vec

        # Initialize encoders
        text_encoder = Text_to_Vec()
        audio_encoder = Audio_to_Vec()
        image_encoder = Image_to_Vec()

        # Encode first sample
        text_vec = text_encoder(batch['text'][0])
        audio_vec = audio_encoder(batch['audio'][0])
        image_vec = image_encoder(batch['video'][0])

        print(f"[OK] All encoders work!")
        print(f"  Text output: {text_vec.shape}")
        print(f"  Audio output: {audio_vec.shape}")
        print(f"  Image output: {image_vec.shape}")

    except Exception as e:
        print(f"[FAIL] Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print("\n" + "=" * 80)
    print("DATALOADER TEST SUMMARY")
    print("=" * 80)
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {min(2, len(dataset))}")
    print(f"Lazy loading: [OK]")
    print(f"Encoder compatibility: [OK]")
    print()
    print("[OK] Training dataloader works with extracted videos!")
    print()
    print("Note: To train on the full MOSI dataset:")
    print("  1. Download all MOSI videos (not just 5 test videos)")
    print("  2. Extract audio + frames for all videos")
    print("  3. Then run training with the full dataset")
    print("=" * 80)


if __name__ == "__main__":
    main()
