"""
Pre-generate encoder tokens for efficient adapter training (FIXED VERSION).

This script processes raw multimodal data through frozen encoders and saves
the resulting tokens to HDF5 files. This enables 10-100x faster adapter training
by eliminating encoder forward passes during training.

Usage:
    python src/Training/pregenerate_tokens_fixed.py

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
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Handle imports with better error messages
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"Error importing PyTorch: {e}")
    print("Please install PyTorch: pip install torch")
    sys.exit(1)

try:
    import h5py
except ImportError as e:
    print(f"Error importing h5py: {e}")
    print("Please install h5py: pip install h5py")
    sys.exit(1)

try:
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Error importing DataLoader: {e}")
    sys.exit(1)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Try importing with better error handling
def safe_import(module_path, class_name):
    """Safely import a module with error handling."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:
        print(f"Warning: Could not import {class_name} from {module_path}: {e}")
        return None

# Import encoders with fallback
try:
    from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
    from src.Encoders.audio.tone_to_vec import Tone_to_Vec
except:
    print("Warning: Could not import encoders directly, trying alternative import")
    Text_to_Vec = safe_import('src.Encoders.text.semantic_to_vec', 'Text_to_Vec')
    Audio_to_Vec = safe_import('src.Encoders.audio.waveform_to_vec', 'Audio_to_Vec')
    Tone_to_Vec = safe_import('src.Encoders.audio.tone_to_vec', 'Tone_to_Vec')
    Image_to_Vec = safe_import('src.Encoders.image.visual_to_vec', 'Image_to_Vec')

try:
    from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
except ImportError as e:
    print(f"Error importing MOSIRawVideoDataset: {e}")
    print("Make sure mosi_dataset.py exists in src/Training/Data_Wrangling/")
    sys.exit(1)

try:
    from PIL import Image as PILImage
except ImportError:
    print("Warning: PIL not found, installing...")
    os.system("pip install Pillow")
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
        if sample.get('audio') is not None:
            audio_waveforms.append(sample['audio'])
        else:
            audio_waveforms.append(np.zeros(16000, dtype=np.float32))
    
    video_frames = []
    for sample in batch:
        if sample.get('video') is not None:
            frame = PILImage.fromarray(sample['video'])
            video_frames.append(frame)
        else:
            video_frames.append(PILImage.new('RGB', (224, 224), color='black'))
    
    segment_ids = [sample.get('segment_id', f'sample_{i}') for i, sample in enumerate(batch)]
    labels = [sample.get('label', 0.0) for sample in batch]
    
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
        
        # Initialize encoders (only those that loaded successfully)
        self.encoders = {}
        
        if Text_to_Vec is not None:
            try:
                self.encoders['text_base'] = Text_to_Vec().to(device)
                print("  ✓ Text encoder loaded")
            except Exception as e:
                print(f"  ✗ Text encoder failed: {e}")
        
        if Audio_to_Vec is not None:
            try:
                self.encoders['audio_waveform'] = Audio_to_Vec().to(device)
                print("  ✓ Audio waveform encoder loaded")
            except Exception as e:
                print(f"  ✗ Audio waveform encoder failed: {e}")
        
        if Tone_to_Vec is not None:
            try:
                self.encoders['audio_tone'] = Tone_to_Vec().to(device)
                print("  ✓ Audio tone encoder loaded")
            except Exception as e:
                print(f"  ✗ Audio tone encoder failed: {e}")
        
        if Image_to_Vec is not None:
            try:
                self.encoders['image_base'] = Image_to_Vec().to(device)
                print("  ✓ Image encoder loaded")
            except Exception as e:
                print(f"  ✗ Image encoder failed: {e}")
        
        # Set to eval mode (no dropout, fixed batch norm)
        for encoder in self.encoders.values():
            encoder.eval()
        
        print(f"Successfully loaded {len(self.encoders)} encoders")
        
        if len(self.encoders) == 0:
            raise RuntimeError("No encoders could be loaded!")
    
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
                # Process each encoder that we have
                if 'text_base' in self.encoders and 'text' in batch:
                    try:
                        text_tokens = self.encoders['text_base'](
                            batch['text'],
                            pregen=True
                        ).cpu().numpy()
                        all_tokens['text_base'].append(text_tokens)
                    except Exception as e:
                        print(f"Warning: Text encoding failed: {e}")
                
                if 'audio_waveform' in self.encoders and 'audio' in batch:
                    try:
                        audio_tokens = self.encoders['audio_waveform'](
                            batch['audio'],
                            pregen=True
                        ).cpu().numpy()
                        all_tokens['audio_waveform'].append(audio_tokens)
                    except Exception as e:
                        print(f"Warning: Audio waveform encoding failed: {e}")
                
                if 'audio_tone' in self.encoders and 'audio' in batch:
                    try:
                        tone_tokens = self.encoders['audio_tone'](
                            batch['audio'],
                            pregen=True
                        ).cpu().numpy()
                        all_tokens['audio_tone'].append(tone_tokens)
                    except Exception as e:
                        print(f"Warning: Audio tone encoding failed: {e}")
                
                if 'image_base' in self.encoders and 'video' in batch:
                    try:
                        image_tokens = self.encoders['image_base'](
                            batch['video'],
                            pregen=True
                        ).cpu().numpy()
                        all_tokens['image_base'].append(image_tokens)
                    except Exception as e:
                        print(f"Warning: Image encoding failed: {e}")
                
                # Store metadata
                all_segment_ids.extend(batch['segment_ids'])
                all_labels.extend(batch['labels'])
        
        # Concatenate all batches
        for name in list(all_tokens.keys()):
            if len(all_tokens[name]) > 0:
                all_tokens[name] = np.vstack(all_tokens[name])
            else:
                # Remove empty encoder results
                del all_tokens[name]
                print(f"Warning: No tokens generated for {name}")
        
        if len(all_tokens) == 0:
            raise RuntimeError("No tokens were generated!")
        
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
            first_encoder = list(all_tokens.keys())[0]
            f.attrs['token_dim'] = all_tokens[first_encoder].shape[1]
            f.attrs['encoders'] = list(all_tokens.keys())
            f.attrs['generated_at'] = datetime.now().isoformat()
        
        # Compute statistics
        metadata = {
            'num_samples': len(all_segment_ids),
            'token_dim': all_tokens[first_encoder].shape[1],
            'encoders': list(all_tokens.keys()),
            'file_size_mb': os.path.getsize(output_path) / (1024 * 1024),
            'tokens_shape': {name: tokens.shape for name, tokens in all_tokens.items()}
        }
        
        return metadata


def main():
    """
    Main function to generate tokens for all dataset splits.
    """
    print("=" * 80)
    print("Token Pre-Generation Pipeline (FIXED)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Data paths:")
    print(f"  MOSI: {MOSI_DATA_PATH}")
    print(f"  Audio: {AUDIO_DIR}")
    print(f"  Video: {VIDEO_DIR}")
    
    # Check if data directories exist
    for name, path in [("MOSI", MOSI_DATA_PATH), ("Audio", AUDIO_DIR), ("Video", VIDEO_DIR)]:
        if not Path(path).exists():
            print(f"Warning: {name} directory not found: {path}")
    
    print("=" * 80)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR) / "mosi"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize token generator
    try:
        generator = TokenGenerator(device=DEVICE)
    except RuntimeError as e:
        print(f"Error initializing generator: {e}")
        return
    
    # Process each split
    metadata = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        try:
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
            
            if len(dataset) == 0:
                print(f"Warning: {split} dataset is empty, skipping...")
                continue
            
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
            
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            import traceback
            traceback.print_exc()
    
    if len(metadata) == 0:
        print("\nError: No tokens were generated!")
        return
    
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
    if 'train' in metadata:
        print(f"  Token dimension: {metadata['train']['token_dim']}")
        print(f"  Encoders: {', '.join(metadata['train']['encoders'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()