"""
Decoders - Semantic Vector to Modality

Organized by target modality:
- text/: Text decoders (semantic to text)
- audio/: Audio decoders (vector to waveform, etc.)
- image/: Image decoders (vector to visual)

Backward-compatible imports for legacy code.
"""

# Text decoders
from .text.vec_to_semantic import Vec_to_Text

# Audio decoders
from .audio.vec_to_waveform import Vec_to_Audio

# Image decoders
from .image.vec_to_visual import Vec_to_Image

# Backward compatibility: maintain old import paths
__all__ = [
    'Vec_to_Text',
    'Vec_to_Audio',
    'Vec_to_Image',
]
