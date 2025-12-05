"""
Unified Collate Functions for Multi-Modal Training

Provides collate function factories for encoder and decoder training pipelines.
Supports lazy loading, feature extraction, and flexible modality selection.

Usage:
    from src.Training.Data_Wrangling.collate_functions import (
        get_encoder_collate_fn,
        get_decoder_collate_fn,
        collate_fn_multi_modal
    )

    # For encoder training (all modalities)
    collate_fn = get_encoder_collate_fn(['text', 'audio', 'video'])

    # For decoder training (specific targets)
    collate_fn = get_decoder_collate_fn(['text', 'audio'], extract_features=True)
"""

import numpy as np
import torch
import librosa
from PIL import Image as PILImage
from typing import List, Dict, Any, Callable, Optional, Union
from dataclasses import dataclass


# Audio settings
AUDIO_SAMPLE_RATE = 16000


@dataclass
class CollateConfig:
    """Configuration for collate function behavior."""
    modalities: List[str] = None  # Which modalities to include
    extract_mel: bool = False  # Extract mel spectrograms for audio
    extract_features: bool = False  # Extract features (vs raw data)
    max_audio_length: int = 160000  # Max audio samples (10s at 16kHz)
    target_image_size: tuple = (224, 224)  # Target image dimensions
    audio_sample_rate: int = AUDIO_SAMPLE_RATE

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['text', 'audio', 'video']


# =============================================================================
# ENCODER COLLATE FUNCTIONS
# =============================================================================

def collate_fn_raw_video(batch: List[Dict]) -> Dict:
    """
    Collate function for raw video/audio data (encoder training).

    Handles eagerly-loaded data from MOSIRawVideoDataset.
    Converts formats as needed by encoders.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched data:
            - 'text': List of raw text strings
            - 'audio': List of audio waveforms (numpy arrays)
            - 'video': List of PIL Images
            - 'segment_id': List of segment identifiers
    """
    # Extract text (raw strings)
    texts = [sample['text'] for sample in batch]

    # Extract audio waveforms
    audio_waveforms = []
    for sample in batch:
        if sample.get('audio') is not None:
            audio_waveforms.append(sample['audio'])
        else:
            # Fallback: 1 second of silence
            audio_waveforms.append(np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32))

    # Convert video frames to PIL Images
    video_frames = []
    for sample in batch:
        if sample.get('video') is not None:
            if isinstance(sample['video'], np.ndarray):
                frame = PILImage.fromarray(sample['video'])
            elif isinstance(sample['video'], PILImage.Image):
                frame = sample['video']
            else:
                frame = PILImage.new('RGB', (224, 224), color='black')
            video_frames.append(frame)
        else:
            video_frames.append(PILImage.new('RGB', (224, 224), color='black'))

    # Extract segment IDs if available
    segment_ids = [sample.get('segment_id', f'sample_{i}') for i, sample in enumerate(batch)]

    return {
        'text': texts,
        'audio': audio_waveforms,
        'video': video_frames,
        'segment_id': segment_ids
    }


def get_encoder_collate_fn(
    modalities: List[str] = None,
    config: Optional[CollateConfig] = None
) -> Callable:
    """
    Factory for encoder training collate functions.

    Args:
        modalities: List of modalities to include ['text', 'audio', 'video']
        config: Optional CollateConfig for detailed settings

    Returns:
        Collate function suitable for DataLoader
    """
    if config is None:
        config = CollateConfig(modalities=modalities)

    def collate_fn(batch: List[Dict]) -> Dict:
        result = {}

        # Text
        if 'text' in config.modalities:
            result['text'] = [sample.get('text', '') for sample in batch]

        # Audio
        if 'audio' in config.modalities:
            audio_list = []
            for sample in batch:
                audio = sample.get('audio')
                if audio is None:
                    audio = np.zeros(config.audio_sample_rate, dtype=np.float32)
                elif isinstance(audio, str):
                    # Path to audio file - load it
                    try:
                        audio, _ = librosa.load(audio, sr=config.audio_sample_rate)
                    except Exception:
                        audio = np.zeros(config.audio_sample_rate, dtype=np.float32)
                audio_list.append(audio)
            result['audio'] = audio_list

        # Video/Image
        if 'video' in config.modalities:
            video_list = []
            for sample in batch:
                video = sample.get('video')
                if video is None:
                    video = PILImage.new('RGB', config.target_image_size, color='black')
                elif isinstance(video, str):
                    # Path to image file - load it
                    try:
                        video = PILImage.open(video).convert('RGB')
                    except Exception:
                        video = PILImage.new('RGB', config.target_image_size, color='black')
                elif isinstance(video, np.ndarray):
                    video = PILImage.fromarray(video)
                video_list.append(video)
            result['video'] = video_list

        # Segment IDs
        result['segment_id'] = [
            sample.get('segment_id', f'sample_{i}')
            for i, sample in enumerate(batch)
        ]

        return result

    return collate_fn


# =============================================================================
# DECODER COLLATE FUNCTIONS
# =============================================================================

def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256
) -> np.ndarray:
    """
    Extract mel spectrogram from audio waveform.

    Args:
        audio: Audio waveform (numpy array)
        sample_rate: Audio sample rate
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Mel spectrogram (time, n_mels)
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Convert to log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T  # (time, n_mels)


def collate_fn_decoder_training(batch: List[Dict]) -> Dict:
    """
    Collate function for decoder training.

    Provides data in formats suitable for decoder training:
    - Text targets for Vec_to_Text
    - Audio waveforms (and optionally mel spectrograms) for Vec_to_Audio
    - Images for Vec_to_Image

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched data suitable for decoder training
    """
    result = {}

    # Text (raw strings for teacher forcing)
    result['text'] = [sample.get('text', '') for sample in batch]

    # Audio waveforms
    audio_list = []
    for sample in batch:
        audio = sample.get('audio')
        if audio is None:
            audio = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
        audio_list.append(audio)
    result['audio'] = audio_list

    # Video/Images (PIL Images)
    video_list = []
    for sample in batch:
        video = sample.get('video')
        if video is None:
            video = PILImage.new('RGB', (224, 224), color='black')
        elif isinstance(video, np.ndarray):
            video = PILImage.fromarray(video)
        video_list.append(video)
    result['video'] = video_list

    # Segment IDs
    result['segment_id'] = [
        sample.get('segment_id', f'sample_{i}')
        for i, sample in enumerate(batch)
    ]

    return result


def collate_fn_decoder_with_mel(batch: List[Dict]) -> Dict:
    """
    Collate function for decoder training with mel spectrogram extraction.

    Includes pre-extracted mel spectrograms for audio decoder training.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched data including mel spectrograms
    """
    result = collate_fn_decoder_training(batch)

    # Extract mel spectrograms
    mel_list = []
    mel_lengths = []
    for audio in result['audio']:
        try:
            mel = extract_mel_spectrogram(audio)
            mel_list.append(mel)
            mel_lengths.append(mel.shape[0])
        except Exception:
            # Fallback: empty mel
            mel_list.append(np.zeros((10, 80), dtype=np.float32))
            mel_lengths.append(10)

    # Pad mel spectrograms to same length
    max_len = max(mel_lengths)
    mel_padded = []
    for mel in mel_list:
        if mel.shape[0] < max_len:
            padding = np.zeros((max_len - mel.shape[0], mel.shape[1]), dtype=mel.dtype)
            mel = np.concatenate([mel, padding], axis=0)
        mel_padded.append(mel)

    result['mel'] = torch.tensor(np.stack(mel_padded), dtype=torch.float32)
    result['mel_lengths'] = torch.tensor(mel_lengths, dtype=torch.long)

    return result


def get_decoder_collate_fn(
    target_modalities: List[str],
    extract_features: bool = False,
    config: Optional[CollateConfig] = None
) -> Callable:
    """
    Factory for decoder training collate functions.

    Args:
        target_modalities: List of target modalities ['text', 'audio', 'image']
        extract_features: Whether to extract features (mel specs, etc.)
        config: Optional CollateConfig for detailed settings

    Returns:
        Collate function suitable for DataLoader
    """
    if extract_features and 'audio' in target_modalities:
        return collate_fn_decoder_with_mel
    else:
        return collate_fn_decoder_training


# =============================================================================
# UNIFIED MULTI-MODAL COLLATE
# =============================================================================

def collate_fn_multi_modal(
    batch: List[Dict],
    config: Optional[CollateConfig] = None
) -> Dict:
    """
    General-purpose multi-modal collate function.

    Handles any combination of modalities with configurable processing.

    Args:
        batch: List of samples from dataset
        config: CollateConfig specifying modalities and processing

    Returns:
        Dictionary with processed multi-modal data
    """
    if config is None:
        config = CollateConfig()

    result = {}

    # Process each requested modality
    for modality in config.modalities:
        if modality == 'text':
            result['text'] = [sample.get('text', '') for sample in batch]

        elif modality == 'audio':
            audio_list = []
            for sample in batch:
                audio = sample.get('audio')
                if audio is None:
                    audio = np.zeros(config.audio_sample_rate, dtype=np.float32)
                elif isinstance(audio, str):
                    try:
                        audio, _ = librosa.load(audio, sr=config.audio_sample_rate)
                    except Exception:
                        audio = np.zeros(config.audio_sample_rate, dtype=np.float32)

                # Truncate or pad to max length
                if len(audio) > config.max_audio_length:
                    audio = audio[:config.max_audio_length]

                audio_list.append(audio)
            result['audio'] = audio_list

            # Optionally extract mel spectrograms
            if config.extract_mel:
                mel_list = []
                for audio in audio_list:
                    try:
                        mel = extract_mel_spectrogram(audio, config.audio_sample_rate)
                        mel_list.append(mel)
                    except Exception:
                        mel_list.append(np.zeros((10, 80), dtype=np.float32))
                result['mel'] = mel_list

        elif modality == 'video':
            video_list = []
            for sample in batch:
                video = sample.get('video')
                if video is None:
                    video = PILImage.new('RGB', config.target_image_size, color='black')
                elif isinstance(video, str):
                    try:
                        video = PILImage.open(video).convert('RGB')
                        video = video.resize(config.target_image_size)
                    except Exception:
                        video = PILImage.new('RGB', config.target_image_size, color='black')
                elif isinstance(video, np.ndarray):
                    video = PILImage.fromarray(video)
                video_list.append(video)
            result['video'] = video_list

    # Always include segment IDs
    result['segment_id'] = [
        sample.get('segment_id', f'sample_{i}')
        for i, sample in enumerate(batch)
    ]

    # Include labels if present
    labels = [sample.get('label') for sample in batch if 'label' in sample]
    if labels:
        result['label'] = labels

    return result


# =============================================================================
# TEXT-ONLY COLLATE (for decoder training)
# =============================================================================

def collate_fn_text_only(batch: List[Dict]) -> List[str]:
    """
    Simple collate function that extracts only text.

    Used for text decoder training where we only need text targets.

    Args:
        batch: List of samples from dataset

    Returns:
        List of text strings
    """
    return [sample.get('text', '') for sample in batch]
