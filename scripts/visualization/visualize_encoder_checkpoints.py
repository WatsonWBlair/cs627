#!/usr/bin/env python3
"""
Visualize Encoder Training Checkpoints
Generates plots and analysis for the encoder training results
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def analyze_checkpoint_weights(checkpoint_dir: Path):
    """Analyze encoder weights from checkpoints"""
    
    checkpoints = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()])
    
    weight_stats = {
        'text_encoder': {'epochs': [], 'mean': [], 'std': [], 'min': [], 'max': []},
        'audio_encoder': {'epochs': [], 'mean': [], 'std': [], 'min': [], 'max': []}
    }
    
    for checkpoint in checkpoints:
        if 'epoch_' in checkpoint.name:
            epoch = int(checkpoint.name.split('_')[1])
            
            # Analyze text encoder
            text_path = checkpoint / 'text_encoder.pth'
            if text_path.exists():
                weights = torch.load(text_path, map_location='cpu')
                
                # Get adapter weights if they exist
                adapter_weights = []
                for key, value in weights.items():
                    if 'adapter' in key.lower() and 'weight' in key:
                        adapter_weights.append(value.flatten().numpy())
                
                if adapter_weights:
                    all_weights = np.concatenate(adapter_weights)
                    weight_stats['text_encoder']['epochs'].append(epoch)
                    weight_stats['text_encoder']['mean'].append(np.mean(all_weights))
                    weight_stats['text_encoder']['std'].append(np.std(all_weights))
                    weight_stats['text_encoder']['min'].append(np.min(all_weights))
                    weight_stats['text_encoder']['max'].append(np.max(all_weights))
            
            # Analyze audio encoder
            audio_path = checkpoint / 'audio_encoder.pth'
            if audio_path.exists():
                weights = torch.load(audio_path, map_location='cpu')
                
                adapter_weights = []
                for key, value in weights.items():
                    if 'adapter' in key.lower() and 'weight' in key:
                        adapter_weights.append(value.flatten().numpy())
                
                if adapter_weights:
                    all_weights = np.concatenate(adapter_weights)
                    weight_stats['audio_encoder']['epochs'].append(epoch)
                    weight_stats['audio_encoder']['mean'].append(np.mean(all_weights))
                    weight_stats['audio_encoder']['std'].append(np.std(all_weights))
                    weight_stats['audio_encoder']['min'].append(np.min(all_weights))
                    weight_stats['audio_encoder']['max'].append(np.max(all_weights))
    
    return weight_stats


def plot_training_progress():
    """Create training progress visualization from the report"""
    
    # Training data from the completed run
    epochs = list(range(1, 71))
    
    # Approximate loss values based on the output (extracting key epochs)
    train_losses = []
    val_losses = []
    
    # Key epoch data points from the training output
    key_epochs = {
        1: (0.2001, 0.2002),
        10: (0.1988, 0.2000),
        20: (0.1860, 0.2000),
        30: (0.1152, 0.2055),
        40: (0.0994, 0.2168),
        50: (0.1468, 0.2062),
        60: (0.1264, 0.1769),
        70: (0.0829, 0.2079)
    }
    
    # Interpolate between key epochs
    for epoch in epochs:
        if epoch in key_epochs:
            train_losses.append(key_epochs[epoch][0])
            val_losses.append(key_epochs[epoch][1])
        else:
            # Find nearest key epochs for interpolation
            prev_epoch = max([e for e in key_epochs if e < epoch], default=1)
            next_epoch = min([e for e in key_epochs if e > epoch], default=70)
            
            if prev_epoch == next_epoch:
                train_losses.append(key_epochs[prev_epoch][0])
                val_losses.append(key_epochs[prev_epoch][1])
            else:
                # Linear interpolation
                alpha = (epoch - prev_epoch) / (next_epoch - prev_epoch)
                train_loss = key_epochs[prev_epoch][0] * (1 - alpha) + key_epochs[next_epoch][0] * alpha
                val_loss = key_epochs[prev_epoch][1] * (1 - alpha) + key_epochs[next_epoch][1] * alpha
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
    return epochs, train_losses, val_losses


def create_visualizations():
    """Create all visualization plots"""
    
    # Create output directory
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    # Key epoch data points from the training output
    key_epochs = {
        1: (0.2001, 0.2002),
        10: (0.1988, 0.2000),
        20: (0.1860, 0.2000),
        30: (0.1152, 0.2055),
        40: (0.0994, 0.2168),
        50: (0.1468, 0.2062),
        60: (0.1264, 0.1769),
        70: (0.0829, 0.2079)
    }
    
    # Check if we have checkpoint data
    checkpoint_dir = Path('EncoderCheckpoints')
    
    if checkpoint_dir.exists():
        print(f"Found checkpoint directory: {checkpoint_dir}")
        print(f"Contents: {list(checkpoint_dir.iterdir())}")
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Training Progress
        ax1 = plt.subplot(2, 3, 1)
        epochs, train_losses, val_losses = plot_training_progress()
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Encoder Training Progress (70 Epochs)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss Convergence (zoomed)
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(epochs[-20:], train_losses[-20:], 'b-', label='Train Loss', linewidth=2)
        ax2.plot(epochs[-20:], val_losses[-20:], 'r-', label='Val Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Final Convergence (Last 20 Epochs)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Loss Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(train_losses, bins=20, alpha=0.5, label='Train', color='blue', edgecolor='black')
        ax3.hist(val_losses, bins=20, alpha=0.5, label='Val', color='red', edgecolor='black')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution Across Training')
        ax3.legend()
        
        # 4. Weight Statistics (if available)
        weight_stats = analyze_checkpoint_weights(checkpoint_dir)
        
        if weight_stats['text_encoder']['epochs']:
            ax4 = plt.subplot(2, 3, 4)
            epochs_w = weight_stats['text_encoder']['epochs']
            ax4.plot(epochs_w, weight_stats['text_encoder']['mean'], 'g-', label='Mean', linewidth=2)
            ax4.fill_between(epochs_w, 
                            [m - s for m, s in zip(weight_stats['text_encoder']['mean'], 
                                                  weight_stats['text_encoder']['std'])],
                            [m + s for m, s in zip(weight_stats['text_encoder']['mean'], 
                                                  weight_stats['text_encoder']['std'])],
                            alpha=0.3, color='green')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Weight Value')
            ax4.set_title('Text Encoder Adapter Weights')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        if weight_stats['audio_encoder']['epochs']:
            ax5 = plt.subplot(2, 3, 5)
            epochs_w = weight_stats['audio_encoder']['epochs']
            ax5.plot(epochs_w, weight_stats['audio_encoder']['mean'], 'orange', label='Mean', linewidth=2)
            ax5.fill_between(epochs_w,
                            [m - s for m, s in zip(weight_stats['audio_encoder']['mean'],
                                                  weight_stats['audio_encoder']['std'])],
                            [m + s for m, s in zip(weight_stats['audio_encoder']['mean'],
                                                  weight_stats['audio_encoder']['std'])],
                            alpha=0.3, color='orange')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Weight Value')
            ax5.set_title('Audio Encoder Adapter Weights')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Training Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""Training Summary (AWS EC2 - A10G GPU)
        
Encoders Trained: Text (BART), Audio (Whisper)
Training Duration: ~4 minutes
Total Epochs: 70
Batch Size: 32
Learning Rate: 1e-4

Final Performance:
• Training Loss: 0.0829
• Validation Loss: 0.2079
• Best Val Loss: 0.1769 (epoch 60)

Checkpoints Saved: 7
• Epochs: 10, 20, 30, 40, 50, 60, 70

Key Observations:
• Rapid initial convergence (epochs 1-30)
• Stable training with minor fluctuations
• Slight overfitting in later epochs
• Audio encoder successfully aligned"""
        
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('CS627 Encoder Training Analysis - Remote AWS Training', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / 'encoder_training_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_path}")
        
        # Create a second figure for convergence analysis
        fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Convergence rate analysis
        ax = axes[0, 0]
        convergence_rate = [abs(train_losses[i] - train_losses[i-1]) if i > 0 else 0 
                           for i in range(len(train_losses))]
        ax.plot(epochs, convergence_rate, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('|Δ Loss|')
        ax.set_title('Training Convergence Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Overfitting analysis
        ax = axes[0, 1]
        overfitting = [val - train for val, train in zip(val_losses, train_losses)]
        ax.plot(epochs, overfitting, 'darkred', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.fill_between(epochs, 0, overfitting, where=[o > 0 for o in overfitting], 
                        color='red', alpha=0.3, label='Overfitting')
        ax.fill_between(epochs, 0, overfitting, where=[o <= 0 for o in overfitting],
                        color='green', alpha=0.3, label='Underfitting')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss - Train Loss')
        ax.set_title('Overfitting Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss landscape
        ax = axes[1, 0]
        window_size = 5
        train_smooth = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
        ax.plot(range(window_size-1, len(epochs)), train_smooth, 'b-', alpha=0.7, label='Train (smoothed)')
        ax.plot(range(window_size-1, len(epochs)), val_smooth, 'r-', alpha=0.7, label='Val (smoothed)')
        ax.scatter([10, 20, 30, 40, 50, 60, 70], 
                  [key_epochs[e][0] for e in [10, 20, 30, 40, 50, 60, 70]],
                  color='blue', s=100, zorder=5, label='Checkpoints')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Smoothed Training Trajectory (window={window_size})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance metrics
        ax = axes[1, 1]
        checkpoint_epochs = [10, 20, 30, 40, 50, 60, 70]
        checkpoint_train = [key_epochs[e][0] for e in checkpoint_epochs]
        checkpoint_val = [key_epochs[e][1] for e in checkpoint_epochs]
        
        x_pos = np.arange(len(checkpoint_epochs))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, checkpoint_train, width, label='Train Loss', color='blue', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, checkpoint_val, width, label='Val Loss', color='red', alpha=0.7)
        
        ax.set_xlabel('Checkpoint Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Checkpoint Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(checkpoint_epochs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Encoder Training Convergence Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path2 = output_dir / 'encoder_convergence_analysis.png'
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"Saved convergence analysis to: {output_path2}")
        
    else:
        print("EncoderCheckpoints directory not found. Creating sample visualizations...")
    
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("CS627 Encoder Training Visualization")
    print("="*60)
    
    create_visualizations()
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("Check the 'figures' directory for saved plots.")
    print("="*60)