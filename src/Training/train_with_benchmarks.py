#!/usr/bin/env python
"""
Enhanced Training Script with Benchmark Data

This script trains encoders using combined data from:
1. CMU-MOSI (multimodal data)
2. MTEB (text retrieval and similarity data)
3. GLUE (text NLI and paraphrase data)

The training uses contrastive learning with mixed batches from all sources,
automatically evaluating on downstream tasks during training.

Usage:
    # Train text encoder with benchmark data
    python src/Training/train_with_benchmarks.py --encoders text --data-sources mteb glue
    
    # Train all encoders with all data
    python src/Training/train_with_benchmarks.py --encoders text audio image --data-sources mosi mteb glue
    
    # Train with automatic evaluation every N epochs
    python src/Training/train_with_benchmarks.py --eval-every 5 --eval-tasks stsb mrpc
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import our modules
from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from src.Training.encoder_trainers import ContrastiveLoss, MomentumEncoder
from src.Training.Data_Wrangling.benchmark_dataset import BenchmarkDataset, collate_fn_benchmark
from src.Evaluation.benchmarks.mteb_adapter import run_mteb_evaluation
from src.Evaluation.benchmarks.glue_adapter import run_glue_evaluation
from src.Evaluation import compute_cross_modal_metrics
from src.Evaluation.visualization import plot_training_curves, plot_semantic_clustering


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train encoders with benchmark data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--encoders',
        nargs='+',
        default=['text'],
        choices=['text', 'audio', 'image'],
        help='Which encoders to train'
    )
    parser.add_argument(
        '--freeze-base',
        action='store_true',
        help='Freeze base model and only train adapters'
    )
    
    # Data configuration
    parser.add_argument(
        '--data-sources',
        nargs='+',
        default=['mteb', 'glue'],
        choices=['mosi', 'mteb', 'glue'],
        help='Which data sources to use for training'
    )
    parser.add_argument(
        '--sampling-weights',
        nargs='+',
        type=float,
        default=None,
        help='Sampling weights for each data source'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per epoch (for debugging)'
    )
    
    # Training configuration
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=500,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--gradient-clip',
        type=float,
        default=1.0,
        help='Gradient clipping value'
    )
    
    # Contrastive learning configuration
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help='Temperature for contrastive loss'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.999,
        help='Momentum for momentum encoder'
    )
    parser.add_argument(
        '--queue-size',
        type=int,
        default=65536,
        help='Queue size for momentum contrast'
    )
    
    # Evaluation configuration
    parser.add_argument(
        '--eval-every',
        type=int,
        default=0,
        help='Evaluate on downstream tasks every N epochs (0 = no eval during training)'
    )
    parser.add_argument(
        '--eval-tasks',
        nargs='+',
        default=['stsb'],
        help='GLUE tasks to evaluate on during training'
    )
    parser.add_argument(
        '--eval-mteb',
        action='store_true',
        help='Also run quick MTEB evaluation'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/benchmark_training/',
        help='Directory to save results'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=5,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='Weights & Biases project name (None = no logging)'
    )
    
    # Data paths
    parser.add_argument(
        '--mosi-data-path',
        type=str,
        default='data/cmumosi/mosi/',
        help='Path to MOSI data'
    )
    parser.add_argument(
        '--mteb-data-path',
        type=str,
        default='data/mteb/',
        help='Path to MTEB wrangled data'
    )
    parser.add_argument(
        '--glue-data-path',
        type=str,
        default='data/glue/',
        help='Path to GLUE wrangled data'
    )
    
    return parser.parse_args()


class BenchmarkTrainer:
    """Trainer for encoders using benchmark datasets."""
    
    def __init__(self, args):
        """Initialize trainer with arguments."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Initialize tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'eval_scores': []
        }
        
        # Initialize wandb if requested
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=f"benchmark_training_{self.timestamp}",
                config=vars(args)
            )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.run_dir / 'tensorboard')
        
    def load_encoders(self) -> Dict[str, nn.Module]:
        """Load and initialize encoders."""
        print("\n" + "="*80)
        print("LOADING ENCODERS")
        print("="*80)
        
        encoders = {}
        
        for name in self.args.encoders:
            if name == 'text':
                print("Loading Text_to_Vec encoder...")
                encoder = Text_to_Vec()
                encoders['text'] = encoder.to(self.device)
                
            elif name == 'audio':
                print("Loading Audio_to_Vec encoder...")
                encoder = Audio_to_Vec()
                encoders['audio'] = encoder.to(self.device)
                
            elif name == 'image':
                print("Loading Image_to_Vec encoder...")
                encoder = Image_to_Vec()
                encoders['image'] = encoder.to(self.device)
            
            print(f"  ✓ {name} encoder loaded")
        
        # Optionally freeze base models
        if self.args.freeze_base:
            print("\nFreezing base models (only training adapters)...")
            for name, encoder in encoders.items():
                # Freeze all parameters except adapter
                for param_name, param in encoder.named_parameters():
                    if 'adapter' not in param_name.lower():
                        param.requires_grad = False
                print(f"  ✓ {name} base model frozen")
        
        return encoders
    
    def load_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Load training and validation data."""
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        
        # Determine which modalities to load
        modalities = []
        if 'text' in self.args.encoders:
            modalities.append('text')
        if 'audio' in self.args.encoders:
            modalities.append('audio')
        if 'image' in self.args.encoders:
            modalities.append('video')  # MOSI uses 'video' for images
        
        # Create training dataset
        print("Loading training data...")
        train_dataset = BenchmarkDataset(
            data_sources=self.args.data_sources,
            split='train',
            modalities=modalities,
            mosi_data_path=self.args.mosi_data_path,
            mteb_data_path=self.args.mteb_data_path,
            glue_data_path=self.args.glue_data_path,
            sampling_weights=self.args.sampling_weights
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn_benchmark,
            pin_memory=True
        )
        
        # Create validation dataset if available
        val_loader = None
        if 'mosi' in self.args.data_sources:
            print("Loading validation data...")
            val_dataset = BenchmarkDataset(
                data_sources=['mosi'],  # Only MOSI has proper val split
                split='valid',
                modalities=modalities,
                mosi_data_path=self.args.mosi_data_path
            )
            
            if len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=collate_fn_benchmark
                )
        
        print(f"  Train batches: {len(train_loader)}")
        if val_loader:
            print(f"  Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_optimizers(self, encoders: Dict[str, nn.Module]) -> Dict:
        """Create optimizers for each encoder."""
        optimizers = {}
        schedulers = {}
        
        for name, encoder in encoders.items():
            # Get trainable parameters
            trainable_params = [p for p in encoder.parameters() if p.requires_grad]
            
            if not trainable_params:
                print(f"Warning: No trainable parameters for {name} encoder")
                continue
            
            # Create optimizer
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
            optimizers[name] = optimizer
            
            # Create scheduler (linear warmup + cosine decay)
            total_steps = self.args.epochs * 1000  # Approximate
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.lr,
                total_steps=total_steps,
                pct_start=0.1
            )
            schedulers[name] = scheduler
        
        return {'optimizers': optimizers, 'schedulers': schedulers}
    
    def train_epoch(
        self,
        epoch: int,
        encoders: Dict[str, nn.Module],
        momentum_encoders: Dict[str, MomentumEncoder],
        train_loader: DataLoader,
        optimizers: Dict,
        criterion: ContrastiveLoss
    ) -> float:
        """Train for one epoch."""
        # Set to train mode
        for encoder in encoders.values():
            encoder.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip if we hit max samples
            if self.args.max_samples and batch_idx * self.args.batch_size >= self.args.max_samples:
                break
            
            batch_loss = 0
            num_modalities = 0
            
            # Process text if available
            if 'text' in encoders and 'anchor_text' in batch:
                anchor_texts = batch['anchor_text']
                positive_texts = batch['positive_text']
                negative_texts = batch['negative_text']
                
                # Encode
                anchor_emb = encoders['text'](anchor_texts)
                positive_emb = encoders['text'](positive_texts)
                negative_emb = encoders['text'](negative_texts)
                
                # Momentum encoding
                with torch.no_grad():
                    mom_positive = momentum_encoders['text'](positive_texts)
                
                # Calculate loss
                loss = criterion(anchor_emb, mom_positive, negative_emb)
                batch_loss += loss
                num_modalities += 1
            
            # Process audio if available
            if 'audio' in encoders and 'anchor_audio' in batch:
                anchor_audio = batch['anchor_audio'].to(self.device)
                positive_audio = batch['positive_audio'].to(self.device)
                negative_audio = batch['negative_audio'].to(self.device)
                
                # Encode
                anchor_emb = encoders['audio'](anchor_audio)
                positive_emb = encoders['audio'](positive_audio)
                negative_emb = encoders['audio'](negative_audio)
                
                # Momentum encoding
                with torch.no_grad():
                    mom_positive = momentum_encoders['audio'].encoder(positive_audio)
                
                # Calculate loss
                loss = criterion(anchor_emb, mom_positive, negative_emb)
                batch_loss += loss
                num_modalities += 1
            
            # Process image if available
            if 'image' in encoders and 'anchor_video' in batch:
                anchor_video = batch['anchor_video'].to(self.device)
                positive_video = batch['positive_video'].to(self.device)
                negative_video = batch['negative_video'].to(self.device)
                
                # Encode
                anchor_emb = encoders['image'](anchor_video)
                positive_emb = encoders['image'](positive_video)
                negative_emb = encoders['image'](negative_video)
                
                # Momentum encoding
                with torch.no_grad():
                    mom_positive = momentum_encoders['image'].encoder(positive_video)
                
                # Calculate loss
                loss = criterion(anchor_emb, mom_positive, negative_emb)
                batch_loss += loss
                num_modalities += 1
            
            # Average loss across modalities
            if num_modalities > 0:
                batch_loss = batch_loss / num_modalities
                
                # Backward pass
                for optimizer in optimizers['optimizers'].values():
                    optimizer.zero_grad()
                
                batch_loss.backward()
                
                # Gradient clipping
                for encoder in encoders.values():
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), self.args.gradient_clip)
                
                # Optimizer step
                for optimizer in optimizers['optimizers'].values():
                    optimizer.step()
                
                # Scheduler step
                for scheduler in optimizers['schedulers'].values():
                    scheduler.step()
                
                # Update momentum encoders
                for name in momentum_encoders:
                    momentum_encoders[name].update()
                
                total_loss += batch_loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': batch_loss.item()})
                
                # Log to wandb
                if self.args.wandb_project:
                    wandb.log({
                        'train_loss': batch_loss.item(),
                        'epoch': epoch,
                        'batch': batch_idx
                    })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def evaluate(
        self,
        encoders: Dict[str, nn.Module],
        val_loader: Optional[DataLoader],
        criterion: ContrastiveLoss
    ) -> Dict:
        """Evaluate encoders."""
        results = {}
        
        # Validation loss if available
        if val_loader:
            for encoder in encoders.values():
                encoder.eval()
            
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch_loss = 0
                    num_modalities = 0
                    
                    # Similar to training but without gradients
                    if 'text' in encoders and 'anchor_text' in batch:
                        anchor_texts = batch['anchor_text']
                        positive_texts = batch['positive_text']
                        negative_texts = batch['negative_text']
                        
                        anchor_emb = encoders['text'](anchor_texts)
                        positive_emb = encoders['text'](positive_texts)
                        negative_emb = encoders['text'](negative_texts)
                        
                        loss = criterion(anchor_emb, positive_emb, negative_emb)
                        batch_loss += loss
                        num_modalities += 1
                    
                    if num_modalities > 0:
                        batch_loss = batch_loss / num_modalities
                        total_loss += batch_loss.item()
                        num_batches += 1
            
            results['val_loss'] = total_loss / num_batches if num_batches > 0 else 0
        
        # Downstream task evaluation
        if self.args.eval_every > 0 and 'text' in encoders:
            print("\nRunning downstream evaluation...")
            
            # GLUE evaluation
            if self.args.eval_tasks:
                from src.Evaluation.benchmarks.glue_adapter import run_glue_evaluation
                glue_results = run_glue_evaluation(
                    encoders['text'],
                    tasks=self.args.eval_tasks,
                    output_dir=self.run_dir / 'glue_eval'
                )
                results['glue'] = glue_results
            
            # MTEB evaluation
            if self.args.eval_mteb:
                from src.Evaluation.benchmarks.mteb_adapter import run_mteb_evaluation
                mteb_results = run_mteb_evaluation(
                    encoders['text'],
                    quick_eval=True,
                    output_dir=self.run_dir / 'mteb_eval'
                )
                results['mteb'] = mteb_results
        
        return results
    
    def save_checkpoint(self, epoch: int, encoders: Dict[str, nn.Module], optimizers: Dict):
        """Save training checkpoint."""
        checkpoint_dir = self.run_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'encoders': {name: enc.state_dict() for name, enc in encoders.items()},
            'optimizers': {name: opt.state_dict() for name, opt in optimizers['optimizers'].items()},
            'history': self.history,
            'args': vars(self.args)
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save adapter weights separately
        for name, encoder in encoders.items():
            # Save adapter weights to OptimalWeights
            adapter_path = Path('OptimalWeights') / f'{name}_adapter_benchmark.pth'
            adapter_path.parent.mkdir(exist_ok=True)
            
            # Extract adapter weights
            adapter_state = {}
            for param_name, param in encoder.state_dict().items():
                if 'adapter' in param_name.lower():
                    adapter_state[param_name] = param
            
            if adapter_state:
                torch.save(adapter_state, adapter_path)
                print(f"Saved {name} adapter weights to {adapter_path}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("BENCHMARK TRAINING")
        print("="*80)
        
        # Load encoders
        encoders = self.load_encoders()
        
        # Create momentum encoders
        momentum_encoders = {}
        for name, encoder in encoders.items():
            momentum_encoders[name] = MomentumEncoder(
                encoder,
                momentum=self.args.momentum,
                queue_size=self.args.queue_size
            )
        
        # Load data
        train_loader, val_loader = self.load_data()
        
        # Create optimizers
        optimizers = self.create_optimizers(encoders)
        
        # Create loss function
        criterion = ContrastiveLoss(temperature=self.args.temperature)
        
        # Training loop
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            # Train
            train_loss = self.train_epoch(
                epoch, encoders, momentum_encoders,
                train_loader, optimizers, criterion
            )
            self.history['train_loss'].append(train_loss)
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs} - Train Loss: {train_loss:.4f}")
            
            # Evaluate
            if (epoch + 1) % self.args.eval_every == 0 or epoch == self.args.epochs - 1:
                eval_results = self.evaluate(encoders, val_loader, criterion)
                
                if 'val_loss' in eval_results:
                    val_loss = eval_results['val_loss']
                    self.history['val_loss'].append(val_loss)
                    print(f"Validation Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(epoch + 1, encoders, optimizers)
                        print("New best model saved!")
                
                # Log downstream results
                if 'glue' in eval_results:
                    glue_avg = eval_results['glue'].get('average', {}).get('score', 0)
                    print(f"GLUE Average: {glue_avg:.3f}")
                    self.history['eval_scores'].append(glue_avg)
                    
                    if self.args.wandb_project:
                        wandb.log({'glue_avg': glue_avg, 'epoch': epoch})
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint(epoch + 1, encoders, optimizers)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if self.history['val_loss']:
                self.writer.add_scalar('Loss/val', self.history['val_loss'][-1], epoch)
        
        # Final save
        self.save_checkpoint(self.args.epochs, encoders, optimizers)
        
        # Generate plots
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80)
        
        # Plot training curves
        fig = plot_training_curves(
            self.history,
            save_path=self.run_dir / 'training_curves.png'
        )
        
        # Save final results
        results_path = self.run_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete! Results saved to {self.run_dir}")
        
        return encoders, self.history


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create trainer
    trainer = BenchmarkTrainer(args)
    
    # Run training
    encoders, history = trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Results saved to: {trainer.run_dir}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    if history['eval_scores']:
        print(f"Final eval score: {history['eval_scores'][-1]:.3f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())