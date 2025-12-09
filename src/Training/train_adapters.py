"""
Adapter-Only Training with Pre-Generated Tokens

Trains adapter MLPs using pre-generated encoder tokens, enabling 10-100x faster
training compared to end-to-end approaches. Supports both encoder and decoder
adapter training with various loss functions.

Usage:
    # Train encoder adapters (contrastive learning)
    python src/Training/train_adapters.py --mode encoder
    
    # Train decoder adapters (reconstruction)
    python src/Training/train_adapters.py --mode decoder

Environment Variables:
    BATCH_SIZE=256: Batch size (can be much larger without encoders)
    LEARNING_RATE=0.001: Learning rate for adapters
    EPOCHS=50: Number of training epochs
    TOKEN_DIR=data/pregenerated_tokens/mosi: Token directory
    ADAPTER_HIDDEN_SIZE=512: Hidden layer size
    ADAPTER_LAYERS=2: Number of hidden layers
    USE_AMP=1: Use automatic mixed precision
    GRADIENT_ACCUMULATION=1: Gradient accumulation steps
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.Adapter import Adapter
from src.Training.Data_Wrangling.token_dataset import create_token_dataloader, load_token_metadata
from src.Training.adapter_trainer import AdapterTrainer, ContrastiveLoss, ReconstructionLoss

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '256'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
EPOCHS = int(os.getenv('EPOCHS', '50'))
TOKEN_DIR = os.getenv('TOKEN_DIR', 'data/pregenerated_tokens/mosi')
ADAPTER_HIDDEN_SIZE = int(os.getenv('ADAPTER_HIDDEN_SIZE', '512'))
ADAPTER_LAYERS = int(os.getenv('ADAPTER_LAYERS', '2'))
USE_AMP = os.getenv('USE_AMP', '1') == '1'
GRADIENT_ACCUMULATION = int(os.getenv('GRADIENT_ACCUMULATION', '1'))

# Output directories
OPTIMAL_WEIGHTS_DIR = os.getenv('OPTIMAL_WEIGHTS_DIR', 'OptimalWeights')
CANDIDATE_WEIGHTS_DIR = os.getenv('CANDIDATE_WEIGHTS_DIR', 'CandidateWeights')
RESULTS_DIR = os.getenv('RESULTS_DIR', 'Results/adapter_training')


class AdapterConfig:
    """Configuration for an adapter."""
    
    def __init__(
        self,
        name: str,
        input_encoder: str,
        output_encoder: Optional[str] = None,
        hidden_size: int = ADAPTER_HIDDEN_SIZE,
        hidden_layers: int = ADAPTER_LAYERS,
        dropout: float = 0.1
    ):
        self.name = name
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder or input_encoder
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        
    def create_adapter(self, input_dim: int = 1024, output_dim: int = 1024) -> Adapter:
        """Create adapter module."""
        return Adapter(
            prefix=self.name,
            input_length=input_dim,
            output_length=output_dim,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            weights_dir=OPTIMAL_WEIGHTS_DIR
        )


def train_encoder_adapters(
    train_loader: DataLoader,
    val_loader: DataLoader,
    adapter_configs: List[AdapterConfig],
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    token_dim: int = 1024
) -> Dict:
    """
    Train encoder adapters using contrastive learning.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        adapter_configs: List of adapter configurations
        epochs: Number of training epochs
        learning_rate: Learning rate
        token_dim: Dimension of input tokens

    Returns:
        Training history and metrics
    """
    print("\n" + "=" * 80)
    print("Training Encoder Adapters (Contrastive Learning)")
    print("=" * 80)

    # Create adapters and encoder mapping
    adapters = {}
    encoder_mapping = {}
    for config in adapter_configs:
        adapter = config.create_adapter(input_dim=token_dim, output_dim=token_dim)
        adapters[config.name] = adapter.to(DEVICE)
        encoder_mapping[config.name] = config.input_encoder
        print(f"Created adapter: {config.name} ({config.input_encoder} -> semantic, dim={token_dim})")

    # Create trainer
    trainer = AdapterTrainer(
        adapters=adapters,
        loss_fn=ContrastiveLoss(temperature=0.07),
        learning_rate=learning_rate,
        device=DEVICE,
        use_amp=USE_AMP,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        encoder_mapping=encoder_mapping
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_recall_at_1': [],
        'val_recall_at_5': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        history['val_loss'].append(val_metrics['loss'])
        history['val_recall_at_1'].append(val_metrics.get('recall@1', 0))
        history['val_recall_at_5'].append(val_metrics.get('recall@5', 0))
        
        # Print progress
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        if 'recall@1' in val_metrics:
            print(f"  Val R@1: {val_metrics['recall@1']:.3f}")
            print(f"  Val R@5: {val_metrics['recall@5']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            trainer.save_adapters()
            print(f"  [BEST] Saved adapters")
        
        # Early stopping
        if epoch - best_epoch > 10:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nBest epoch: {best_epoch} with val loss: {best_val_loss:.4f}")
    
    return history


def train_decoder_adapters(
    train_loader: DataLoader,
    val_loader: DataLoader,
    adapter_configs: List[AdapterConfig],
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    token_dim: int = 1024
) -> Dict:
    """
    Train decoder adapters using reconstruction loss.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        adapter_configs: List of adapter configurations
        epochs: Number of training epochs
        learning_rate: Learning rate
        token_dim: Dimension of input tokens

    Returns:
        Training history and metrics
    """
    print("\n" + "=" * 80)
    print("Training Decoder Adapters (Reconstruction)")
    print("=" * 80)

    # Create adapters
    adapters = {}
    for config in adapter_configs:
        adapter = config.create_adapter(input_dim=token_dim, output_dim=token_dim)
        adapters[config.name] = adapter.to(DEVICE)
        print(f"Created adapter: {config.name} (semantic -> {config.output_encoder}, dim={token_dim})")
    
    # Create trainer
    trainer = AdapterTrainer(
        adapters=adapters,
        loss_fn=ReconstructionLoss(),
        learning_rate=learning_rate,
        device=DEVICE,
        use_amp=USE_AMP,
        gradient_accumulation=GRADIENT_ACCUMULATION
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_cosine_sim': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        history['val_loss'].append(val_metrics['loss'])
        history['val_cosine_sim'].append(val_metrics.get('cosine_similarity', 0))
        
        # Print progress
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        if 'cosine_similarity' in val_metrics:
            print(f"  Val Cosine Sim: {val_metrics['cosine_similarity']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            trainer.save_adapters()
            print(f"  [BEST] Saved adapters")
        
        # Early stopping
        if epoch - best_epoch > 10:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nBest epoch: {best_epoch} with val loss: {best_val_loss:.4f}")
    
    return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train adapters with pre-generated tokens')
    parser.add_argument('--mode', type=str, default='encoder', choices=['encoder', 'decoder'],
                        help='Training mode: encoder (contrastive) or decoder (reconstruction)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--token-dir', type=str, default=TOKEN_DIR,
                        help='Directory containing pre-generated tokens')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Adapter Training with Pre-Generated Tokens")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Token directory: {args.token_dir}")
    print(f"Mixed precision: {USE_AMP}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print("=" * 80)
    
    # Load token metadata
    try:
        metadata = load_token_metadata(args.token_dir)
        print(f"\nLoaded token metadata:")
        print(f"  Train samples: {metadata['train']['num_samples']}")
        print(f"  Val samples: {metadata['valid']['num_samples']}")
        print(f"  Token dimension: {metadata['train']['token_dim']}")
        print(f"  Available encoders: {', '.join(metadata['train']['encoders'])}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please run pregenerate_tokens.py first to generate tokens.")
        return
    
    # Create data loaders
    print("\nCreating data loaders...")
    
    if args.mode == 'encoder':
        # For encoder training, use all tokens in contrastive mode
        train_loader = create_token_dataloader(
            'train',
            token_dir=args.token_dir,
            batch_size=args.batch_size,
            shuffle=True,
            token_mode='matched',
            cache_size=2000
        )
        
        val_loader = create_token_dataloader(
            'valid',
            token_dir=args.token_dir,
            batch_size=args.batch_size,
            shuffle=False,
            token_mode='matched',
            cache_size=1000
        )
        
        # Define adapter configurations
        adapter_configs = [
            AdapterConfig('text_adapter', 'text_base'),
            AdapterConfig('audio_adapter', 'audio_waveform'),
            AdapterConfig('tone_adapter', 'audio_tone'),
            AdapterConfig('image_adapter', 'image_base')
        ]
        
        # Train encoder adapters
        token_dim = metadata['train']['token_dim']
        history = train_encoder_adapters(
            train_loader,
            val_loader,
            adapter_configs,
            epochs=args.epochs,
            learning_rate=args.lr,
            token_dim=token_dim
        )
        
    else:  # decoder mode
        # For decoder training, use text tokens as source
        train_loader = create_token_dataloader(
            'train',
            token_dir=args.token_dir,
            batch_size=args.batch_size,
            shuffle=True,
            token_mode='single',
            primary_encoder='text_base',
            cache_size=2000
        )
        
        val_loader = create_token_dataloader(
            'valid',
            token_dir=args.token_dir,
            batch_size=args.batch_size,
            shuffle=False,
            token_mode='single',
            primary_encoder='text_base',
            cache_size=1000
        )
        
        # Define adapter configurations for decoders
        adapter_configs = [
            AdapterConfig('text_decoder_adapter', 'text_base', 'text_base'),
            AdapterConfig('audio_decoder_adapter', 'text_base', 'audio_waveform'),
            AdapterConfig('image_decoder_adapter', 'text_base', 'image_base')
        ]
        
        # Train decoder adapters
        token_dim = metadata['train']['token_dim']
        history = train_decoder_adapters(
            train_loader,
            val_loader,
            adapter_configs,
            epochs=args.epochs,
            learning_rate=args.lr,
            token_dim=token_dim
        )
    
    # Save training history
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_path = results_dir / f"{args.mode}_training_{timestamp}.json"
    
    with open(history_path, 'w') as f:
        json.dump({
            'mode': args.mode,
            'config': {
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'epochs': args.epochs,
                'adapter_hidden_size': ADAPTER_HIDDEN_SIZE,
                'adapter_layers': ADAPTER_LAYERS
            },
            'history': history,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\nTraining complete! History saved to: {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()