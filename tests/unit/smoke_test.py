"""
Smoke tests for production-ready encoders.

Quick sanity checks for critical functionality:
- Encoder initialization
- Input format validation
- Output shape validation
- Adapter loading/saving
"""

import sys
import os
import unittest
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from Encoders.text_2_vec import Text_to_Vec
from Encoders.wav_2_vec import Audio_to_Vec
from Encoders.img_2_vec import Image_to_Vec


class TestTextToVec(unittest.TestCase):
    """Smoke test Text_to_Vec encoder (PRODUCTION)."""

    def setUp(self):
        """Initialize encoder before each test."""
        self.encoder = Text_to_Vec()

    def test_initialization(self):
        """Test encoder initializes correctly."""
        self.assertIsNotNone(self.encoder)
        self.assertTrue(hasattr(self.encoder, 'adapter'))

    def test_string_input(self):
        """Test encoder accepts string input."""
        text = "This is a test sentence for the encoder."
        output = self.encoder(text)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1024))

    def test_empty_string(self):
        """Test encoder handles empty string."""
        text = ""
        output = self.encoder(text)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1024))


class TestAudioToVec(unittest.TestCase):
    """Smoke test Audio_to_Vec encoder (PRODUCTION)."""

    def setUp(self):
        """Initialize encoder before each test."""
        self.encoder = Audio_to_Vec()

    def test_initialization(self):
        """Test encoder initializes correctly."""
        self.assertIsNotNone(self.encoder)
        self.assertTrue(hasattr(self.encoder, 'adapter'))

    def test_waveform_input(self):
        """Test encoder accepts waveform input."""
        # Create dummy 1-second waveform at 16kHz
        waveform = np.random.randn(16000).astype(np.float32)
        output = self.encoder(waveform)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1024))


class TestImageToVec(unittest.TestCase):
    """Smoke test Image_to_Vec encoder (PRODUCTION)."""

    def setUp(self):
        """Initialize encoder before each test."""
        self.encoder = Image_to_Vec()

    def test_initialization(self):
        """Test encoder initializes correctly."""
        self.assertIsNotNone(self.encoder)
        self.assertTrue(hasattr(self.encoder, 'adapter'))

    def test_pil_image_input(self):
        """Test encoder accepts PIL Image input."""
        # Create dummy RGB image
        image = Image.new('RGB', (224, 224), color='red')
        output = self.encoder(image)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1024))


class TestEncoderConsistency(unittest.TestCase):
    """Smoke test consistency across encoders."""

    def setUp(self):
        """Initialize all encoders."""
        self.text_encoder = Text_to_Vec()
        self.audio_encoder = Audio_to_Vec()
        self.image_encoder = Image_to_Vec()

    def test_output_dimension_consistency(self):
        """Test all encoders output same dimension."""
        text = "Test text"
        waveform = np.random.randn(16000).astype(np.float32)
        image = Image.new('RGB', (224, 224), color='green')

        text_output = self.text_encoder(text)
        audio_output = self.audio_encoder(waveform)
        image_output = self.image_encoder(image)

        # All should output (1, 1024)
        self.assertEqual(text_output.shape, (1, 1024))
        self.assertEqual(audio_output.shape, (1, 1024))
        self.assertEqual(image_output.shape, (1, 1024))


if __name__ == '__main__':
    unittest.main()
