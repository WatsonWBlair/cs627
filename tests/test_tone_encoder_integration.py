"""
Quick integration test for tone encoder in training pipeline.
Tests:
1. All encoders can be imported
2. EncoderConfig works with loss_weight parameter
3. All 4 encoders initialize correctly
"""

import sys
import os
sys.path.insert(0, 'src')

from Training.train_encoders import EncoderConfig
from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from Encoders.audio.tone_to_vec import Tone_to_Vec

print("=" * 80)
print("Tone Encoder Integration Test")
print("=" * 80)

print("\n[Step 1/3] Testing imports...")
print("[OK] All imports successful")

print("\n[Step 2/3] Creating 4-encoder swarm configs...")
encoder_configs = [
    EncoderConfig(
        name='text_base',
        encoder=Text_to_Vec(),
        data_key='text',
        loss_weight=1.0
    ),
    EncoderConfig(
        name='audio_waveform',
        encoder=Audio_to_Vec(),
        data_key='audio',
        loss_weight=1.0
    ),
    EncoderConfig(
        name='audio_tone',
        encoder=Tone_to_Vec(),
        data_key='audio',
        loss_weight=1.0
    ),
    EncoderConfig(
        name='image_base',
        encoder=Image_to_Vec(),
        data_key='video',
        loss_weight=1.0
    ),
]

print(f"[OK] Created {len(encoder_configs)} encoder configs")
for cfg in encoder_configs:
    print(f"  - {cfg.name}: data_key='{cfg.data_key}', loss_weight={cfg.loss_weight}")

print("\n[Step 3/3] Testing encoder forward passes...")
import numpy as np
import torch
from PIL import Image

# Test text encoder
text_vec = encoder_configs[0].encoder("Test sentence")
print(f"[OK] text_base: {text_vec.shape}")
assert text_vec.shape == (1, 1024)

# Test audio waveform encoder
test_audio = np.random.randn(16000)
audio_waveform_vec = encoder_configs[1].encoder(test_audio)
print(f"[OK] audio_waveform: {audio_waveform_vec.shape}")
assert audio_waveform_vec.shape == (1, 1024)

# Test audio tone encoder (same input!)
audio_tone_vec = encoder_configs[2].encoder(test_audio)
print(f"[OK] audio_tone: {audio_tone_vec.shape}")
assert audio_tone_vec.shape == (1, 1024)

# Test image encoder
test_image = Image.new('RGB', (224, 224), color='red')
image_vec = encoder_configs[3].encoder(test_image)
print(f"[OK] image_base: {image_vec.shape}")
assert image_vec.shape == (1, 1024)

print("\n" + "=" * 80)
print("SUCCESS: All 4 encoders working correctly!")
print("=" * 80)
print("\nEncoder swarm ready for training with:")
print("  - Feature importance weighting (loss_weight parameter)")
print("  - Multiple specialized encoders per modality")
print("  - audio_waveform (Whisper) + audio_tone (WavLM) from same audio input")
