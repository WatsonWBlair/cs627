"""
Standalone Cross-Modal Encoder Training
Works without mmsdk dependency - uses raw audio/video files directly
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Any
import copy
import time
import json
from datetime import datetime
from pathlib import Path
import random

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
import librosa
from PIL import Image as PILImage

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters (configurable via environment variables)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0001'))
EPOCHS = int(os.getenv('EPOCHS', '5'))
MOMENTUM = float(os.getenv('MOMENTUM', '0.999'))
QUEUE_SIZE = int(os.getenv('QUEUE_SIZE', '2048'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.07'))

# Data paths
AUDIO_DIR = os.getenv('AUDIO_DIR', 'data/cmumosi/audio/')
VIDEO_DIR = os.getenv('VIDEO_DIR', 'data/cmumosi/frames/')


class SimpleMultimodalDataset(Dataset):
    """Simple dataset that loads audio/video pairs directly from disk."""
    
    def __init__(self, audio_dir: str, video_dir: str, max_samples: int = None):
        """
        Initialize dataset with audio and video directories.
        
        Args:
            audio_dir: Directory containing .wav files
            video_dir: Directory containing .jpg files
            max_samples: Maximum number of samples to use (for testing)
        """
        self.audio_dir = Path(audio_dir)
        self.video_dir = Path(video_dir)
        
        # Get list of files (assuming matching names)
        audio_files = sorted(list(self.audio_dir.glob("*.wav")))
        video_files = sorted(list(self.video_dir.glob("*.jpg")))
        
        # Match files by stem (filename without extension)
        self.samples = []
        video_stems = {f.stem: f for f in video_files}
        
        for audio_file in audio_files:
            stem = audio_file.stem
            if stem in video_stems:
                self.samples.append({
                    'audio': audio_file,
                    'video': video_stems[stem],
                    'text': f"Segment {stem}"  # Placeholder text
                })
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} multimodal samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio (16kHz)
        audio, _ = librosa.load(sample['audio'], sr=16000, mono=True)
        
        # Load image
        image = PILImage.open(sample['video']).convert('RGB')
        
        # Text (placeholder or could load from file)
        text = sample['text']
        
        return {
            'audio': audio.astype(np.float32),
            'video': image,
            'text': text
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    return {
        'text': [item['text'] for item in batch],
        'audio': [item['audio'] for item in batch],
        'video': [item['video'] for item in batch]
    }


class SimpleContrastiveLoss(nn.Module):
    """Simple contrastive loss for training."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss between all modality pairs.
        
        Args:
            embeddings: Dict mapping modality names to embeddings
            
        Returns:
            Loss value
        """
        losses = []
        modalities = list(embeddings.keys())
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                emb1 = F.normalize(embeddings[modalities[i]], dim=1)
                emb2 = F.normalize(embeddings[modalities[j]], dim=1)
                
                # Cosine similarity
                sim = torch.matmul(emb1, emb2.t()) / self.temperature
                
                # Contrastive loss (diagonal elements are positive pairs)
                labels = torch.arange(sim.shape[0]).to(sim.device)
                loss_i = F.cross_entropy(sim, labels)
                loss_j = F.cross_entropy(sim.t(), labels)
                
                losses.append((loss_i + loss_j) / 2)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)


def train_epoch(
    encoders: Dict[str, nn.Module],
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """Train for one epoch."""
    
    for encoder in encoders.values():
        encoder.train()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Encode each modality
        embeddings = {}
        
        try:
            # Text encoding
            if 'text' in encoders:
                embeddings['text'] = encoders['text'](batch['text'], device=device)
            
            # Audio encoding
            if 'audio' in encoders:
                embeddings['audio'] = encoders['audio'](batch['audio'], device=device)
            
            # Image encoding
            if 'image' in encoders:
                embeddings['image'] = encoders['image'](batch['video'], device=device)
            
            # Compute loss
            loss = criterion(embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Warning: CUDA out of memory, skipping batch")
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return total_loss / max(num_batches, 1)


def main():
    """Main training function."""
    print("=" * 80)
    print("Standalone Cross-Modal Encoder Training")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"Video dir: {VIDEO_DIR}")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = SimpleMultimodalDataset(
        audio_dir=AUDIO_DIR,
        video_dir=VIDEO_DIR,
        max_samples=100  # Limit for testing
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize encoders
    print("\n[2/4] Initializing encoders...")
    encoders = {}
    
    try:
        encoders['text'] = Text_to_Vec().to(DEVICE)
        print("✓ Text encoder loaded")
        # Try to load existing adapter weights
        if hasattr(encoders['text'], 'adapter'):
            weights_path = encoders['text'].adapter.weights_path
            if os.path.exists(weights_path):
                encoders['text'].adapter.load()
                print(f"  ✓ Loaded existing text adapter weights from {weights_path}")
            else:
                print(f"  ℹ No existing text adapter weights found, starting fresh")
    except Exception as e:
        print(f"Warning: Could not load text encoder: {e}")
    
    try:
        encoders['audio'] = Audio_to_Vec().to(DEVICE)
        print("✓ Audio encoder loaded")
        # Try to load existing adapter weights
        if hasattr(encoders['audio'], 'adapter'):
            weights_path = encoders['audio'].adapter.weights_path
            if os.path.exists(weights_path):
                encoders['audio'].adapter.load()
                print(f"  ✓ Loaded existing audio adapter weights from {weights_path}")
            else:
                print(f"  ℹ No existing audio adapter weights found, starting fresh")
    except Exception as e:
        print(f"Warning: Could not load audio encoder: {e}")
    
    # Note: Image encoder might fail with PyTorch < 2.6
    try:
        encoders['image'] = Image_to_Vec().to(DEVICE)
        print("✓ Image encoder loaded")
        # Try to load existing adapter weights
        if hasattr(encoders['image'], 'adapter'):
            weights_path = encoders['image'].adapter.weights_path
            if os.path.exists(weights_path):
                encoders['image'].adapter.load()
                print(f"  ✓ Loaded existing image adapter weights from {weights_path}")
            else:
                print(f"  ℹ No existing image adapter weights found, starting fresh")
    except Exception as e:
        print(f"Warning: Could not load image encoder: {e}")
    
    if len(encoders) < 2:
        print("Error: Need at least 2 encoders for contrastive learning")
        return
    
    # Create optimizer (only for adapters)
    print("\n[3/4] Creating optimizer...")
    params = []
    for name, encoder in encoders.items():
        if hasattr(encoder, 'adapter'):
            params.extend(encoder.adapter.parameters())
            # Freeze pretrained parts
            if hasattr(encoder, 'encoder'):
                for param in encoder.encoder.parameters():
                    param.requires_grad = False
    
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = SimpleContrastiveLoss(temperature=TEMPERATURE)
    
    # Check if we're resuming training
    start_epoch = 1
    checkpoint_file = "training_checkpoint.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_epoch = checkpoint.get('last_epoch', 0) + 1
                print(f"ℹ Resuming from epoch {start_epoch}")
        except:
            pass
    
    # Training loop
    print("\n[4/4] Starting training...")
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)
        
        avg_loss = train_epoch(
            encoders=encoders,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )
        
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save adapters
        if epoch % 5 == 0 or epoch == EPOCHS:
            print("Saving adapter weights...")
            for name, encoder in encoders.items():
                if hasattr(encoder, 'adapter'):
                    try:
                        encoder.adapter.save()
                        print(f"✓ Saved {name} adapter")
                    except Exception as e:
                        print(f"Warning: Could not save {name} adapter: {e}")
        
        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'last_epoch': epoch,
                'last_loss': avg_loss,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Clear cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    # Final memory stats
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"Final GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")


if __name__ == "__main__":
    main()