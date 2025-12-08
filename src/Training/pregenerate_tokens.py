"""
Pre-generate encoder tokens for efficient adapter training.

This script processes raw multimodal data through frozen encoders and saves
the resulting tokens to HDF5 files. This enables 10-100x faster adapter training
by eliminating encoder forward passes during training.

Usage:
    python src/Training/pregenerate_tokens.py

Environment Variables:
    BATCH_SIZE=32: Batch size for token generation
    MOSI_DATA_PATH=data/cmumosi/mosi/: Path to MOSI metadata
    AUDIO_DIR=data/cmumosi/audio/: Extracted audio files
    VIDEO_DIR=data/cmumosi/frames/: Extracted video frames
    OUTPUT_DIR=data/pregenerated_tokens/: Output directory for tokens
    DEVICE=cuda: Device for encoding (cuda/cpu)
"""

import os
import sys
import h5py
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from src.Encoders.audio.tone_to_vec import Tone_to_Vec
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
from PIL import Image as PILImage

# Configuration
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'data/pregenerated_tokens')
MOSI_DATA_PATH = os.getenv('MOSI_DATA_PATH', 'data/cmumosi/mosi/')
AUDIO_DIR = os.getenv('AUDIO_DIR', 'data/cmumosi/audio/')
VIDEO_DIR = os.getenv('VIDEO_DIR', 'data/cmumosi/frames/')

# Dataset split ratios
TRAIN_RATIO = float(os.getenv('TRAIN_RATIO', '0.7'))
VAL_RATIO = float(os.getenv('VAL_RATIO', '0.15'))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))


def collate_fn_pregeneration(batch: List[Dict]) -> Dict:
    """
    Collate function for pre-generation.
    Prepares batch data for encoder processing.
    """
    texts = [sample['text'] for sample in batch]
    
    audio_waveforms = []
    for sample in batch:
        if sample['audio'] is not None:
            audio_waveforms.append(sample['audio'])
        else:
            audio_waveforms.append(np.zeros(16000, dtype=np.float32))
    
    video_frames = []
    for sample in batch:
        if sample['video'] is not None:
            frame = PILImage.fromarray(sample['video'])
            video_frames.append(frame)
        else:
            video_frames.append(PILImage.new('RGB', (224, 224), color='black'))
    
    segment_ids = [sample['segment_id'] for sample in batch]
    labels = [sample['label'] for sample in batch]
    
    return {
        'text': texts,
        'audio': audio_waveforms,
        'video': video_frames,
        'segment_ids': segment_ids,
        'labels': labels
    }


class TokenGenerator:
    """
    Generates and saves encoder tokens for a dataset.
    """
    
    def __init__(self, device: str = DEVICE):
        """
        Initialize encoders for token generation.
        """
        self.device = device
        print(f"Initializing TokenGenerator on {device}")
        
        # Initialize encoders
        self.encoders = {
            'text_base': Text_to_Vec().to(device),
            'audio_waveform': Audio_to_Vec().to(device),
            'audio_tone': Tone_to_Vec().to(device),
            'image_base': Image_to_Vec().to(device)
        }
        
        # Set to eval mode (no dropout, fixed batch norm)
        for encoder in self.encoders.values():
            encoder.eval()
        
        print(f"Loaded {len(self.encoders)} encoders:")
        for name in self.encoders.keys():
            print(f"  - {name}")
    
    def generate_tokens(
        self,
        dataloader: DataLoader,
        output_path: str,
        desc: str = "Generating tokens"
    ) -> Dict:
        """
        Generate tokens for a dataset and save to HDF5.
        
        Args:
            dataloader: DataLoader for the dataset
            output_path: Path to save HDF5 file
            desc: Description for progress bar
        
        Returns:
            Metadata dictionary with statistics
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # First pass: collect all tokens in memory
        all_tokens = {name: [] for name in self.encoders.keys()}
        all_segment_ids = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # Text encoding
                text_tokens = self.encoders['text_base'](
                    batch['text'],
                    pregen=True
                ).cpu().numpy()
                all_tokens['text_base'].append(text_tokens)
                
                # Audio waveform encoding
                audio_tokens = self.encoders['audio_waveform'](
                    batch['audio'],
                    pregen=True
                ).cpu().numpy()
                all_tokens['audio_waveform'].append(audio_tokens)
                
                # Audio tone encoding
                tone_tokens = self.encoders['audio_tone'](
                    batch['audio'],
                    pregen=True
                ).cpu().numpy()
                all_tokens['audio_tone'].append(tone_tokens)
                
                # Image encoding
                image_tokens = self.encoders['image_base'](
                    batch['video'],
                    pregen=True
                ).cpu().numpy()
                all_tokens['image_base'].append(image_tokens)
                
                # Store metadata
                all_segment_ids.extend(batch['segment_ids'])
                all_labels.extend(batch['labels'])
        
        # Concatenate all batches
        for name in all_tokens.keys():
            all_tokens[name] = np.vstack(all_tokens[name])
        
        # Save to HDF5
        print(f"Saving tokens to {output_path}")
        with h5py.File(output_path, 'w') as f:
            # Save tokens
            for name, tokens in all_tokens.items():
                f.create_dataset(
                    name,
                    data=tokens,
                    compression='gzip',
                    compression_opts=4
                )
            
            # Save metadata
            f.create_dataset(
                'segment_ids',
                data=[s.encode('utf-8') for s in all_segment_ids]
            )
            f.create_dataset(
                'labels',
                data=np.array(all_labels, dtype=np.float32)
            )
            
            # Save attributes
            f.attrs['num_samples'] = len(all_segment_ids)
            f.attrs['token_dim'] = all_tokens['text_base'].shape[1]
            f.attrs['encoders'] = list(self.encoders.keys())
            f.attrs['generated_at'] = datetime.now().isoformat()
        
        # Compute statistics
        metadata = {
            'num_samples': len(all_segment_ids),
            'token_dim': all_tokens['text_base'].shape[1],
            'encoders': list(self.encoders.keys()),
            'file_size_mb': os.path.getsize(output_path) / (1024 * 1024),
            'tokens_shape': {name: tokens.shape for name, tokens in all_tokens.items()}
        }
        
        return metadata


def main():
    """
    Main function to generate tokens for all dataset splits.
    """
    print("=" * 80)
    print("Token Pre-Generation Pipeline")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Data paths:")
    print(f"  MOSI: {MOSI_DATA_PATH}")
    print(f"  Audio: {AUDIO_DIR}")
    print(f"  Video: {VIDEO_DIR}")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR) / "mosi"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize token generator
    generator = TokenGenerator(device=DEVICE)
    
    # Process each split
    metadata = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Load dataset
        dataset = MOSIRawVideoDataset(
            split=split,
            mosi_data_path=MOSI_DATA_PATH,
            audio_dir=AUDIO_DIR,
            video_dir=VIDEO_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=RANDOM_SEED
        )
        
        print(f"Loaded {len(dataset)} {split} samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_pregeneration,
            num_workers=0
        )
        
        # Generate tokens
        split_metadata = generator.generate_tokens(
            dataloader,
            str(output_path / f"{split}_tokens.h5"),
            desc=f"Generating {split} tokens"
        )
        
        metadata[split] = split_metadata
        print(f"Generated {split_metadata['num_samples']} samples")
        print(f"File size: {split_metadata['file_size_mb']:.2f} MB")
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Token generation complete!")
    print(f"Tokens saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print summary
    total_samples = sum(m['num_samples'] for m in metadata.values())
    total_size = sum(m['file_size_mb'] for m in metadata.values())
    
    print(f"\nSummary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total size: {total_size:.2f} MB")
    print(f"  Token dimension: {metadata['train']['token_dim']}")
    print(f"  Encoders: {', '.join(metadata['train']['encoders'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()