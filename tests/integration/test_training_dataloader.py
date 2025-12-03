"""
Test the training dataloader with lazy loading collate function.

Validates that the dataloader can:
1. Load dataset successfully
2. Create batches with lazy loading
3. Pass data to encoders correctly
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
    """
    Lazy loading collate function for raw video/audio data.
    """
    # Extract text (raw strings)
    texts = [sample['text'] for sample in batch]

    # Lazy load audio waveforms (16kHz mono for Whisper)
    audio_waveforms = []
    for sample in batch:
        try:
            waveform, sr = librosa.load(sample['audio'], sr=16000, mono=True)
            audio_waveforms.append(waveform)
        except Exception as e:
            print(f"[WARNING] Failed to load audio {sample['audio']}: {e}")
            audio_waveforms.append(np.zeros(16000, dtype=np.float32))

    # Lazy load video frames (RGB PIL Images for ViT)
    video_frames = []
    for sample in batch:
        try:
            frame = PILImage.open(sample['video']).convert('RGB')
            video_frames.append(frame)
        except Exception as e:
            print(f"[WARNING] Failed to load frame {sample['video']}: {e}")
            video_frames.append(PILImage.new('RGB', (224, 224), color='black'))

    return {
        'text': texts,
        'audio': audio_waveforms,
        'video': video_frames
    }


def main():
    print("=" * 80)
    print("Testing Training Dataloader with Lazy Loading")
    print("=" * 80)

    # Load dataset
    print("\n[Step 1/4] Loading dataset...")
    print("-" * 80)

    try:
        dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/',
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42
        )
        print(f"[OK] Loaded {len(dataset)} training samples")

        if len(dataset) == 0:
            print("[ERROR] Dataset is empty!")
            return

    except Exception as e:
        print(f"[FAIL] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create dataloader
    print("\n[Step 2/4] Creating dataloader...")
    print("-" * 80)

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,  # Small batch for testing
            shuffle=False,
            collate_fn=collate_fn_raw_video,
            num_workers=0
        )
        print(f"[OK] Created dataloader with batch_size=2")

    except Exception as e:
        print(f"[FAIL] Failed to create dataloader: {e}")
        return

    # Test loading a batch
    print("\n[Step 3/4] Loading first batch...")
    print("-" * 80)

    try:
        batch = next(iter(dataloader))
        print(f"[OK] Loaded batch successfully")
        print(f"\nBatch contents:")
        print(f"  Text samples: {len(batch['text'])}")
        print(f"  Audio samples: {len(batch['audio'])}")
        print(f"  Video samples: {len(batch['video'])}")
        print(f"\nSample 0:")
        print(f"  Text: '{batch['text'][0][:80]}...'")
        print(f"  Audio shape: {batch['audio'][0].shape}")
        print(f"  Audio duration: {len(batch['audio'][0])/16000:.2f}s")
        print(f"  Video frame: {batch['video'][0].size} {batch['video'][0].mode}")

    except Exception as e:
        print(f"[FAIL] Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with encoders
    print("\n[Step 4/4] Testing with encoders...")
    print("-" * 80)

    try:
        from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec

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
    print(f"Batch size: 2")
    print(f"Lazy loading: [OK]")
    print(f"Encoder compatibility: [OK]")
    print()
    print("[OK] Training dataloader is ready!")
    print("=" * 80)


if __name__ == "__main__":
    main()
