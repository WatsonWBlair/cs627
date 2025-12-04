"""
Data extraction utilities for MOSI dataset.

This package contains utilities for extracting audio segments and video frames
from downloaded MOSI videos using timestamp information from the MOSI dataset.
"""

from .mosi_audio_extractor import (
    parse_segment_id,
    get_segment_timestamps,
    extract_audio_segment,
    extract_audio_segments
)

from .mosi_frame_extractor import (
    extract_frame,
    extract_frames
)

__all__ = [
    'parse_segment_id',
    'get_segment_timestamps',
    'extract_audio_segment',
    'extract_audio_segments',
    'extract_frame',
    'extract_frames'
]
