"""
Encoders - Modality to Semantic Vector

Organized by source modality:
- text/: Text encoders (semantic features)
- audio/: Audio encoders (waveform, tone, etc.)
- image/: Image encoders (visual features)
- video/: Video encoders (motion, temporal features)

Backward-compatible imports for legacy code.
"""

# Text encoders
from .text.semantic_to_vec import Text_to_Vec

# Audio encoders
from .audio.waveform_to_vec import Audio_to_Vec
from .audio.tone_to_vec import Tone_to_Vec

# Image encoders
from .image.visual_to_vec import Image_to_Vec

# Backward compatibility: maintain old import paths
__all__ = [
    'Text_to_Vec',
    'Audio_to_Vec',
    'Tone_to_Vec',
    'Image_to_Vec',
]
