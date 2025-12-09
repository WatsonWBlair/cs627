#!/usr/bin/env python3
"""
Generate demo tokens for ablation study Docker image.
Creates small, representative token files for testing and demos.
"""

import h5py
import numpy as np
import json
from pathlib import Path

def generate_demo_tokens(output_dir: str = "data/pregenerated_tokens/mosi"):
    """Generate demo token files for ablation studies."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating demo tokens for ablation studies...")
    
    # Configuration
    token_dim = 1024
    encoders = ['text_base', 'audio_waveform', 'audio_tone', 'image_base']
    
    # Dataset sizes (small for demo)
    splits = {
        'train': 500,   # Reduced from 1547
        'val': 100,     # Reduced from 331
        'test': 100     # Reduced from 331
    }
    
    # Generate tokens for each split
    for split_name, n_samples in splits.items():
        print(f"  Generating {split_name} tokens: {n_samples} samples")
        
        # Create HDF5 file
        h5_path = output_path / f"{split_name}_tokens.h5"
        with h5py.File(h5_path, 'w') as f:
            # Generate tokens for each encoder
            for encoder in encoders:
                # Generate random tokens (normally distributed)
                # In practice, these would come from real encoders
                tokens = np.random.randn(n_samples, token_dim).astype(np.float32)
                
                # Normalize to unit sphere (like real encoder outputs)
                tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
                
                # Save to HDF5 with compression
                f.create_dataset(
                    encoder,
                    data=tokens,
                    compression='gzip',
                    compression_opts=4
                )
            
            # Add metadata
            segment_ids = [f"{split_name}_{i:04d}" for i in range(n_samples)]
            f.create_dataset(
                'segment_ids',
                data=[s.encode('utf-8') for s in segment_ids]
            )
            
            # Add dummy labels (sentiment scores)
            labels = np.random.uniform(-3, 3, n_samples).astype(np.float32)
            f.create_dataset('labels', data=labels)
            
            # Set attributes
            f.attrs['num_samples'] = n_samples
            f.attrs['token_dim'] = token_dim
            f.attrs['encoders'] = encoders
            f.attrs['type'] = 'demo'
            f.attrs['description'] = 'Demo tokens for ablation studies'
        
        # Print file size
        file_size_mb = h5_path.stat().st_size / (1024 * 1024)
        print(f"    Created {h5_path.name}: {file_size_mb:.1f} MB")
    
    # Create metadata file
    metadata = {
        'description': 'Demo tokens for ablation study Docker image',
        'token_dim': token_dim,
        'encoders': encoders,
        'splits': splits,
        'total_samples': sum(splits.values()),
        'type': 'demo'
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Demo tokens generated in: {output_path}")
    print(f"  Total samples: {sum(splits.values())}")
    print(f"  Total size: ~{sum([p.stat().st_size for p in output_path.glob('*.h5')]) / (1024*1024):.1f} MB")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    # Allow custom output directory
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pregenerated_tokens/mosi"
    generate_demo_tokens(output_dir)