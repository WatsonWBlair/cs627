#!/usr/bin/env python3
"""
Quick Training Test
Minimal test to verify training setup works without MOSI SDK
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_training_setup():
    """Test that we can load encoders and run a forward pass."""
    
    print("=" * 60)
    print("Quick Training Setup Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test Text Encoder
    print("\nTesting Text Encoder...")
    try:
        from Encoders import Text_to_Vec
        text_encoder = Text_to_Vec().to(device)
        
        # Test forward pass
        test_text = ["This is a test sentence", "Another test"]
        output = text_encoder(test_text, device=device)
        print(f"✓ Text encoder output shape: {output.shape}")
        assert output.shape == (2, 1024), f"Expected shape (2, 1024), got {output.shape}"
        del text_encoder
        torch.cuda.empty_cache() if device == "cuda" else None
    except Exception as e:
        print(f"✗ Text encoder failed: {e}")
        return False
    
    # Test Audio Encoder
    print("\nTesting Audio Encoder...")
    try:
        from Encoders import Audio_to_Vec
        audio_encoder = Audio_to_Vec().to(device)
        
        # Test forward pass with dummy audio
        test_audio = [np.random.randn(16000), np.random.randn(16000)]  # 1 second at 16kHz
        output = audio_encoder(test_audio, device=device)
        print(f"✓ Audio encoder output shape: {output.shape}")
        assert output.shape == (2, 1024), f"Expected shape (2, 1024), got {output.shape}"
        del audio_encoder
        torch.cuda.empty_cache() if device == "cuda" else None
    except Exception as e:
        print(f"✗ Audio encoder failed: {e}")
        return False
    
    # Test Memory
    if device == "cuda":
        print("\nGPU Memory Status:")
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Cached: {cached:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✓ Training setup test PASSED!")
    print("Instance is ready for training.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_training_setup()
    sys.exit(0 if success else 1)