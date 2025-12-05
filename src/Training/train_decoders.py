"""
Unified Cross-Modal Decoder Training

Trains N decoders jointly to convert semantic vectors back to their target modalities.
Mirrors the encoder training architecture (train_encoders.py) for consistency.

Features:
- Simultaneous training of multiple decoders (text, audio, image)
- Configurable semantic vector sources (text_only, matched, mixed)
- Two-stage training for audio decoder (proxy -> generation)
- Semantic fidelity monitoring (not in loss function)
- Single optimizer for all decoder adapters

Usage:
    python src/Training/train_decoders.py

    # Environment variable configuration:
    SEMANTIC_SOURCE=mixed python src/Training/train_decoders.py
    TRAIN_TEXT=1 TRAIN_AUDIO=1 python src/Training/train_decoders.py

Based on: src/Training/train_encoders.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from src.Decoders import Vec_to_Text
from src.Training.decoder_trainer import DecoderConfig, CrossModalDecoderTrainer, DecoderTrainingReporter
from src.Training.decoder_losses import (
    compute_text_decoder_loss,
    compute_audio_decoder_loss,
    compute_image_decoder_loss
)
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
from src.Training.Data_Wrangling.collate_functions import collate_fn_decoder_training

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# HYPERPARAMETERS (Configurable via environment variables)
# =============================================================================

BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0001'))
EPOCHS = int(os.getenv('EPOCHS', '20'))
SEMANTIC_DIM = int(os.getenv('SEMANTIC_DIM', '1024'))

# Semantic source configuration
SEMANTIC_SOURCE = os.getenv('SEMANTIC_SOURCE', 'text_only')  # 'text_only', 'matched', 'mixed'

# Which decoders to train
TRAIN_TEXT = os.getenv('TRAIN_TEXT', '1') == '1'
TRAIN_AUDIO = os.getenv('TRAIN_AUDIO', '0') == '1'  # Disabled by default (experimental)
TRAIN_IMAGE = os.getenv('TRAIN_IMAGE', '0') == '1'  # Disabled by default (experimental)

# Audio decoder two-stage training
AUDIO_PROXY_EPOCHS = int(os.getenv('AUDIO_PROXY_EPOCHS', '15'))
AUDIO_GENERATION_EPOCHS = int(os.getenv('AUDIO_GENERATION_EPOCHS', '5'))

# Data paths
MOSI_DATA_PATH = os.getenv('MOSI_DATA_PATH', 'data/cmumosi/mosi/')
AUDIO_DIR = os.getenv('AUDIO_DIR', 'data/cmumosi/audio/')
VIDEO_DIR = os.getenv('VIDEO_DIR', 'data/cmumosi/frames/')

# Training settings
TRAIN_RATIO = float(os.getenv('TRAIN_RATIO', '0.7'))
VAL_RATIO = float(os.getenv('VAL_RATIO', '0.15'))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '0'))

# Evaluation settings
EVAL_FIDELITY_EVERY = int(os.getenv('EVAL_FIDELITY_EVERY', '50'))
SAVE_EVERY = int(os.getenv('SAVE_EVERY', '5'))

# Output directories
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', 'DecoderCheckpoints/')
REPORT_DIR = os.getenv('REPORT_DIR', 'training_reports/decoders/')


def main():
    """Main training function."""
    print("=" * 80)
    print("UNIFIED CROSS-MODAL DECODER TRAINING")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Semantic source: {SEMANTIC_SOURCE}")
    print(f"Decoders to train: " +
          f"{'Text ' if TRAIN_TEXT else ''}" +
          f"{'Audio ' if TRAIN_AUDIO else ''}" +
          f"{'Image ' if TRAIN_IMAGE else ''}")
    print(f"Data paths: MOSI={MOSI_DATA_PATH}")
    print("=" * 80)

    # =========================================================================
    # 1. LOAD SOURCE ENCODERS (frozen, for semantic vector generation)
    # =========================================================================

    print("\n[1/6] Loading source encoders...")

    source_encoders = {}

    # Text encoder (always needed for text_only and mixed modes)
    print("  Loading Text_to_Vec...")
    source_encoders['text'] = Text_to_Vec()
    print("  [OK] Text encoder loaded")

    # Audio encoder (for matched and mixed modes)
    if SEMANTIC_SOURCE in ['matched', 'mixed'] or TRAIN_AUDIO:
        try:
            print("  Loading Audio_to_Vec...")
            source_encoders['audio'] = Audio_to_Vec()
            print("  [OK] Audio encoder loaded")
        except Exception as e:
            print(f"  [WARNING] Audio encoder not available: {e}")

    # Image encoder (for matched and mixed modes)
    if SEMANTIC_SOURCE in ['matched', 'mixed'] or TRAIN_IMAGE:
        try:
            print("  Loading Image_to_Vec...")
            source_encoders['image'] = Image_to_Vec()
            print("  [OK] Image encoder loaded")
        except Exception as e:
            print(f"  [WARNING] Image encoder not available: {e}")

    print(f"  Source encoders: {list(source_encoders.keys())}")

    # =========================================================================
    # 2. LOAD DECODERS
    # =========================================================================

    print("\n[2/6] Loading decoders...")

    decoder_configs = []

    # Text decoder
    if TRAIN_TEXT:
        try:
            print("  Loading Vec_to_Text...")
            text_decoder = Vec_to_Text()
            decoder_configs.append(DecoderConfig(
                name='text_base',
                decoder=text_decoder,
                target_modality='text',
                loss_fn=compute_text_decoder_loss,
                loss_weight=1.0,
                training_stage='full'
            ))
            print("  [OK] Text decoder loaded")
        except Exception as e:
            print(f"  [ERROR] Text decoder failed: {e}")
            TRAIN_TEXT = False

    # Audio decoder
    if TRAIN_AUDIO:
        try:
            print("  Loading Vec_to_Audio...")
            from src.Decoders import Vec_to_Audio
            audio_decoder = Vec_to_Audio()
            decoder_configs.append(DecoderConfig(
                name='audio_waveform',
                decoder=audio_decoder,
                target_modality='audio',
                loss_fn=compute_audio_decoder_loss,
                loss_weight=1.0,
                training_stage='proxy'  # Start with proxy stage
            ))
            print("  [OK] Audio decoder loaded (stage: proxy)")
        except Exception as e:
            print(f"  [WARNING] Audio decoder not available: {e}")
            print("  [WARNING] Skipping audio decoder training")

    # Image decoder
    if TRAIN_IMAGE:
        try:
            print("  Loading Vec_to_Image...")
            from src.Decoders import Vec_to_Image
            image_decoder = Vec_to_Image()
            decoder_configs.append(DecoderConfig(
                name='image_base',
                decoder=image_decoder,
                target_modality='image',
                loss_fn=compute_image_decoder_loss,
                loss_weight=1.0,
                training_stage='full'
            ))
            print("  [OK] Image decoder loaded")
        except Exception as e:
            print(f"  [WARNING] Image decoder not available: {e}")
            print("  [WARNING] Skipping image decoder training")

    if not decoder_configs:
        print("\n[ERROR] No decoders loaded! Nothing to train.")
        return 1

    print(f"\n  Total decoders: {len(decoder_configs)}")

    # =========================================================================
    # 3. LOAD DATASET
    # =========================================================================

    print("\n[3/6] Loading dataset...")

    try:
        train_dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path=MOSI_DATA_PATH,
            audio_dir=AUDIO_DIR,
            video_dir=VIDEO_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=RANDOM_SEED
        )
        print(f"  [OK] Loaded {len(train_dataset)} training samples")

        if len(train_dataset) == 0:
            print("  [ERROR] Dataset is empty!")
            return 1

    except (FileNotFoundError, OSError) as e:
        print(f"  [ERROR] Dataset files not found: {e}")
        print("  Please run data extraction scripts:")
        print("    python scripts/data_wrangling/download_all_mosi_videos.py")
        print("    python scripts/data_wrangling/extract_all_segments.py")
        return 1

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_decoder_training,
        num_workers=NUM_WORKERS
    )

    print(f"  [OK] Batches per epoch: {len(train_loader)}")

    # =========================================================================
    # 4. CREATE TRAINER
    # =========================================================================

    print("\n[4/6] Creating trainer...")

    trainer = CrossModalDecoderTrainer(
        decoder_configs=decoder_configs,
        source_encoders=source_encoders,
        reference_encoders=source_encoders,  # Same encoders for fidelity monitoring
        semantic_source=SEMANTIC_SOURCE,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    print("  [OK] Trainer created")

    # Create reporter
    decoder_names = [config.name for config in decoder_configs]
    reporter = DecoderTrainingReporter(
        decoder_names=decoder_names,
        output_dir=REPORT_DIR
    )

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # =========================================================================
    # 5. TRAINING LOOP
    # =========================================================================

    print("\n[5/6] Starting training...")
    reporter.start_training()

    best_loss = float('inf')
    training_history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{EPOCHS}")

        # Check if we need to switch audio training stage
        # Two-stage audio training: proxy first, then generation
        for config in decoder_configs:
            if config.name == 'audio_waveform':
                if epoch == AUDIO_PROXY_EPOCHS + 1:
                    trainer.update_training_stage('audio_waveform', 'generation')
                    print(f"  [STAGE CHANGE] Audio decoder: proxy -> generation")

        print(f"{'=' * 80}")

        epoch_start = time.time()

        # Train for one epoch
        avg_losses, avg_fidelity = trainer.train_epoch(
            train_loader,
            epoch,
            eval_fidelity_every=EVAL_FIDELITY_EVERY
        )

        epoch_duration = time.time() - epoch_start

        # Record metrics
        reporter.record_epoch(epoch, avg_losses, avg_fidelity, epoch_duration)

        # Print metrics
        print(f"\nEpoch {epoch} Results:")
        for name in sorted(avg_losses.keys()):
            if name != 'total_loss':
                print(f"  {name:20s}: {avg_losses[name]:.4f}")
        print(f"  {'total_loss':20s}: {avg_losses['total_loss']:.4f}")

        if avg_fidelity:
            print("\nSemantic Fidelity (monitoring):")
            for name, score in avg_fidelity.items():
                print(f"  {name:20s}: {score:.4f}")

        print(f"  Epoch Time: {reporter.format_duration(epoch_duration)}")

        # Save best model
        if avg_losses['total_loss'] < best_loss:
            best_loss = avg_losses['total_loss']
            print(f"\n  [OK] New best loss: {best_loss:.4f}")
            trainer.save_all_adapters()

        # Periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.json")
            checkpoint_data = {
                'epoch': epoch,
                'losses': avg_losses,
                'fidelity': avg_fidelity,
                'best_loss': best_loss
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"  [OK] Checkpoint saved: {checkpoint_path}")

        # Record history
        training_history.append({
            'epoch': epoch,
            'losses': avg_losses,
            'fidelity': avg_fidelity,
            'duration': epoch_duration
        })

    # =========================================================================
    # 6. TRAINING COMPLETE
    # =========================================================================

    reporter.end_training()

    print("\n[6/6] Training complete!")

    # Final save
    print("\nSaving final adapter weights...")
    trainer.save_all_adapters()

    # Save training configuration
    config_path = os.path.join(CHECKPOINT_DIR, "training_config.json")
    training_config = {
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'semantic_source': SEMANTIC_SOURCE,
        },
        'decoders': [
            {
                'name': config.name,
                'target_modality': config.target_modality,
                'loss_weight': config.loss_weight,
                'training_stage': config.training_stage
            }
            for config in decoder_configs
        ],
        'source_encoders': list(source_encoders.keys()),
        'device': DEVICE,
        'training_date': datetime.now().isoformat(),
        'best_loss': best_loss
    }

    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    print(f"Training configuration saved: {config_path}")

    # Save training history
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({'history': training_history}, f, indent=2)
    print(f"Training history saved: {history_path}")

    # Save reporter metrics
    reporter.save_metrics()
    reporter.print_summary()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest loss: {best_loss:.4f}")
    print("\nAdapter weights saved:")
    for config in decoder_configs:
        decoder = trainer.decoders[config.name]
        if hasattr(decoder, 'adapter'):
            print(f"  - {config.name}: {decoder.adapter.weights_path}")

    print(f"\nReports saved to: {REPORT_DIR}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")

    print("\nNext steps:")
    print("  1. Review training reports")
    print("  2. Run evaluation: python scripts/evaluate_decoder.py")
    print("  3. Test inference: python -c \"from src.Decoders import Vec_to_Text; ...\"")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
