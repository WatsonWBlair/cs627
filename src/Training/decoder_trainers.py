"""
Text Decoder Adapter Training - Hybrid Approach

Trains Vec_to_Text adapter using teacher forcing with semantic fidelity monitoring.

Training Objective:
    Train adapter to map semantic vectors to BART-compatible representations
    that enable high-quality text generation with minimal semantic drift.

Training Flow (Hybrid Approach):
1. Input text → Text_to_Vec (frozen) → semantic_vec
2. semantic_vec → Adapter (trainable) → adapter_output
3. Use adapter_output as BART encoder_outputs
4. Teacher forcing: BART decoder with target labels → cross-entropy loss (differentiable)
5. Periodic evaluation: Generate text, re-encode, measure semantic drift (monitoring only)
6. Backprop through adapter using teacher forcing loss

This combines:
- Direct text generation optimization (teacher forcing)
- Semantic drift measurement (Fig. 4 from paper)

Usage:
    python src/Training/train_text_decoder.py

Based on: "Bootstrapping Multi-Modal Semantic-Vector Spaces" (Fig. 4, page 3)
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import time
from datetime import datetime
import json

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.Encoders import Text_to_Vec
from src.Decoders import Vec_to_Text
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset

# ============================================================================
# HYPERPARAMETERS (Configurable)
# ============================================================================

# Training parameters
BATCH_SIZE = 16        # Small for fast iteration, increase if GPU allows
LEARNING_RATE = 1e-4   # Standard for adapter training
EPOCHS = 20            # Increase if underfitting

# Evaluation frequency
EVAL_SEMANTIC_EVERY = 10  # Evaluate semantic drift every N batches (0 = only at epoch end)

# Dataset paths
MOSI_DATA_PATH = "data/cmumosi/mosi/"
AUDIO_DIR = "data/cmumosi/audio/"
VIDEO_DIR = "data/cmumosi/frames/"

# Checkpointing
CHECKPOINT_DIR = "DecoderCheckpoints/text/"
SAVE_EVERY = 5  # Save checkpoint every N epochs

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATA LOADING
# ============================================================================

def collate_fn_text_only(batch):
    """
    Extract only text from MOSI dataset batch.

    MOSI dataset returns: {'text': str, 'audio': path, 'video': path, ...}
    We only need the 'text' field for decoder training.

    Args:
        batch: List of MOSI dataset samples

    Returns:
        List[str]: Text strings only
    """
    return [sample['text'] for sample in batch]


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_teacher_forcing_loss(
    semantic_vectors: torch.Tensor,
    target_texts: List[str],
    adapter,
    bart_model,
    tokenizer,
    device: str
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute teacher forcing loss for decoder training.

    Flow:
    1. semantic_vec → adapter → adapter_output (batch, 768)
    2. Create encoder_outputs with adapter_output
    3. Forward through BART decoder with target text labels
    4. Compute cross-entropy loss (differentiable through adapter)

    Args:
        semantic_vectors: (batch, 1024) semantic vectors from Text_to_Vec
        target_texts: Ground truth text strings
        adapter: Trainable adapter (1024 → 768)
        bart_model: BART model for decoding
        tokenizer: BART tokenizer
        device: Device to run on

    Returns:
        (loss, metrics_dict)
    """
    from transformers.modeling_outputs import BaseModelOutput

    # 1. Map semantic vectors to BART space
    adapter_output = adapter(semantic_vectors)  # (batch, 768)

    # 2. Create encoder outputs for BART decoder
    # Note: Using single vector (batch, 1, 768) as encoder output
    # Future: Support sequences (batch, seq_len, 768)
    encoder_outputs = BaseModelOutput(
        last_hidden_state=adapter_output.unsqueeze(1),  # (batch, 1, 768)
        hidden_states=None,
        attentions=None
    )

    # 3. Tokenize target texts for teacher forcing
    target_encoding = tokenizer(
        target_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # 4. Forward through BART decoder with labels (teacher forcing)
    outputs = bart_model(
        encoder_outputs=encoder_outputs,
        labels=target_encoding.input_ids
    )

    # Cross-entropy loss (automatically computed when labels provided)
    loss = outputs.loss

    return loss, {
        'teacher_forcing_loss': loss.item()
    }


def evaluate_semantic_fidelity(
    semantic_vectors: torch.Tensor,
    decoder,
    encoder,
    device: str
) -> Dict[str, float]:
    """
    Evaluate semantic drift after decode-encode cycle.

    This is NON-DIFFERENTIABLE (uses .generate()) - for monitoring only.

    Flow:
    1. semantic_vec → decoder.generate() → text
    2. text → encoder → reconstructed_vec
    3. Measure cosine similarity (semantic_vec, reconstructed_vec)

    Args:
        semantic_vectors: (batch, 1024) original semantic vectors
        decoder: Vec_to_Text decoder
        encoder: Text_to_Vec encoder
        device: Device to run on

    Returns:
        Dict with semantic fidelity metrics
    """
    with torch.no_grad():
        # Generate text from semantic vectors
        reconstructed_texts = decoder(semantic_vectors, device=device)

        # Handle single vs batch output
        if isinstance(reconstructed_texts, str):
            reconstructed_texts = [reconstructed_texts]

        # Re-encode generated text
        reconstructed_vecs = encoder(reconstructed_texts, device=device)

        # Compute cosine similarity
        original_norm = F.normalize(semantic_vectors, dim=1)
        reconstructed_norm = F.normalize(reconstructed_vecs, dim=1)
        cosine_sim = F.cosine_similarity(original_norm, reconstructed_norm, dim=1)

        return {
            'semantic_similarity': cosine_sim.mean().item(),
            'semantic_drift': 1.0 - cosine_sim.mean().item()  # Lower is better
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    train_loader,
    decoder,
    encoder,
    bart_model,
    tokenizer,
    optimizer,
    epoch: int,
    eval_semantic_every: int
) -> Dict[str, float]:
    """
    Train decoder adapter for one epoch using hybrid approach.

    Args:
        train_loader: DataLoader with text data
        decoder: Vec_to_Text decoder (adapter trainable)
        encoder: Text_to_Vec encoder (frozen)
        bart_model: BART model for teacher forcing (frozen)
        tokenizer: BART tokenizer
        optimizer: Adam optimizer for decoder adapter
        epoch: Current epoch number
        eval_semantic_every: Evaluate semantic drift every N batches (0 = only at end)

    Returns:
        Dict with average metrics for the epoch
    """
    decoder.adapter.train()  # Only adapter is trainable
    encoder.eval()
    bart_model.eval()

    epoch_metrics = {
        'teacher_forcing_loss': 0.0,
        'semantic_similarity': 0.0,
        'semantic_drift': 0.0
    }
    num_batches = 0
    semantic_eval_count = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch_idx, batch_texts in enumerate(pbar, 1):
        # 1. Encode texts to semantic vectors (frozen)
        with torch.no_grad():
            semantic_vecs = encoder(batch_texts, device=DEVICE)

        # 2. Compute teacher forcing loss (differentiable)
        loss, metrics = compute_teacher_forcing_loss(
            semantic_vecs,
            batch_texts,
            decoder.adapter,
            bart_model,
            tokenizer,
            DEVICE
        )

        # 3. Backprop through adapter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        epoch_metrics['teacher_forcing_loss'] += metrics['teacher_forcing_loss']
        num_batches += 1

        # 4. Periodically evaluate semantic fidelity (non-differentiable, monitoring only)
        should_eval_semantic = (
            eval_semantic_every > 0 and batch_idx % eval_semantic_every == 0
        )

        if should_eval_semantic:
            semantic_metrics = evaluate_semantic_fidelity(
                semantic_vecs, decoder, encoder, DEVICE
            )
            epoch_metrics['semantic_similarity'] += semantic_metrics['semantic_similarity']
            epoch_metrics['semantic_drift'] += semantic_metrics['semantic_drift']
            semantic_eval_count += 1

            # Update progress bar with semantic metrics
            pbar.set_postfix({
                'tf_loss': f"{metrics['teacher_forcing_loss']:.4f}",
                'sem_sim': f"{semantic_metrics['semantic_similarity']:.3f}",
                'drift': f"{semantic_metrics['semantic_drift']:.3f}"
            })
        else:
            # Show only training loss
            pbar.set_postfix({
                'tf_loss': f"{metrics['teacher_forcing_loss']:.4f}"
            })

    # Average metrics over all batches
    epoch_metrics['teacher_forcing_loss'] /= num_batches
    if semantic_eval_count > 0:
        epoch_metrics['semantic_similarity'] /= semantic_eval_count
        epoch_metrics['semantic_drift'] /= semantic_eval_count
    else:
        # Evaluate once at end of epoch if not evaluated during training
        print("\n  Evaluating semantic fidelity at epoch end...")
        # Use first batch for quick evaluation
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_vecs = encoder(sample_batch, device=DEVICE)
        semantic_metrics = evaluate_semantic_fidelity(
            sample_vecs, decoder, encoder, DEVICE
        )
        epoch_metrics['semantic_similarity'] = semantic_metrics['semantic_similarity']
        epoch_metrics['semantic_drift'] = semantic_metrics['semantic_drift']

    return epoch_metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""
    print("=" * 80)
    print("TEXT DECODER ADAPTER TRAINING - HYBRID APPROACH")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Semantic eval frequency: every {EVAL_SEMANTIC_EVERY} batches" if EVAL_SEMANTIC_EVERY > 0 else "Semantic eval frequency: once per epoch")
    print("=" * 80)
    print()
    print("Training Strategy:")
    print("  Primary loss: Teacher forcing (cross-entropy, differentiable)")
    print("  Monitoring: Semantic drift (cosine similarity, non-differentiable)")
    print("=" * 80)
    print()

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ========================================================================
    # 1. LOAD MODELS
    # ========================================================================

    print("[1/4] Loading models...")
    print()

    # Load encoder (frozen for training)
    print("  Loading Text_to_Vec encoder...")
    encoder = Text_to_Vec().to(DEVICE)
    encoder.eval()

    # Freeze all encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False

    print("  [OK] Text_to_Vec encoder: LOADED & FROZEN")
    print()

    # Load decoder (only adapter trainable)
    print("  Loading Vec_to_Text decoder...")
    decoder = Vec_to_Text().to(DEVICE)

    # Freeze pretrained BART decoder, only train adapter
    for param in decoder.decoder.parameters():
        param.requires_grad = False

    print("  [OK] Vec_to_Text decoder: LOADED (adapter trainable)")
    print(f"  [OK] Adapter path: {decoder.adapter.weights_path}")
    print()

    # Load BART model for ground truth hidden states
    print("  Loading BART model for target hidden states...")
    from transformers import BartForConditionalGeneration, BartTokenizer
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(DEVICE)
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart_model.eval()

    for param in bart_model.parameters():
        param.requires_grad = False

    print("  [OK] BART model: LOADED & FROZEN")
    print()

    # Create optimizer (only adapter parameters)
    optimizer = torch.optim.Adam(decoder.adapter.parameters(), lr=LEARNING_RATE)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in decoder.adapter.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # ========================================================================
    # 2. LOAD DATASET
    # ========================================================================

    print("[2/4] Loading dataset...")
    print()

    try:
        train_dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path=MOSI_DATA_PATH,
            audio_dir=AUDIO_DIR,
            video_dir=VIDEO_DIR,
            required_modalities=['text']  # Text-only training for decoder
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_text_only,
            num_workers=0  # Set to 0 for Windows compatibility
        )

        print(f"  [OK] Training samples: {len(train_dataset)}")
        print(f"  [OK] Batches per epoch: {len(train_loader)}")
        print()

    except FileNotFoundError as e:
        print(f"  [FAIL] Dataset not found: {e}")
        print(f"  Please run data preparation scripts first:")
        print(f"    1. python scripts/data_wrangling/download_all_mosi_videos.py")
        print(f"    2. python scripts/data_wrangling/extract_test_segments.py")
        return 1

    # ========================================================================
    # 3. TRAINING LOOP
    # ========================================================================

    print("[3/4] Training...")
    print()

    best_loss = float('inf')
    training_history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"{'=' * 80}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'=' * 80}")

        epoch_start = time.time()

        # Train for one epoch
        metrics = train_epoch(
            train_loader, decoder, encoder, bart_model, bart_tokenizer,
            optimizer, epoch, EVAL_SEMANTIC_EVERY
        )

        epoch_duration = time.time() - epoch_start

        # Log metrics
        print()
        print(f"Epoch {epoch} Results:")
        print(f"  Teacher Forcing Loss: {metrics['teacher_forcing_loss']:.4f}")
        print(f"  Semantic Similarity:  {metrics['semantic_similarity']:.3f}")
        print(f"  Semantic Drift:       {metrics['semantic_drift']:.3f}")
        print(f"  Epoch Time:           {epoch_duration:.1f}s")

        # Save best model (based on teacher forcing loss)
        if metrics['teacher_forcing_loss'] < best_loss:
            best_loss = metrics['teacher_forcing_loss']
            decoder.adapter.save()
            print(f"  [OK] SAVED (best loss: {best_loss:.4f})")

        print()

        # Save checkpoint every N epochs
        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'adapter_state_dict': decoder.adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'metrics': metrics
            }, checkpoint_path)
            print(f"  [OK] Checkpoint saved: {checkpoint_path}")
            print()

        # Record history
        training_history.append({
            'epoch': epoch,
            'metrics': metrics,
            'duration': epoch_duration
        })

    # ========================================================================
    # 4. TRAINING COMPLETE
    # ========================================================================

    total_duration = time.time() - start_time

    print()
    print("=" * 80)
    print("[4/4] Training Complete!")
    print("=" * 80)
    print()
    print(f"Total training time: {total_duration / 60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final adapter weights: {decoder.adapter.weights_path}")
    print()

    # Save training history
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'hyperparameters': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'eval_semantic_every': EVAL_SEMANTIC_EVERY,
                'training_approach': 'hybrid_teacher_forcing'
            },
            'history': training_history,
            'best_loss': best_loss,
            'total_duration': total_duration
        }, f, indent=2)

    print(f"Training history saved: {history_path}")
    print()

    # Print training summary
    print("Training Summary:")
    print(f"  Initial loss: {training_history[0]['metrics']['total_loss']:.4f}")
    print(f"  Final loss:   {training_history[-1]['metrics']['total_loss']:.4f}")
    print(f"  Best loss:    {best_loss:.4f}")
    print(f"  Improvement:  {training_history[0]['metrics']['total_loss'] - best_loss:.4f}")
    print()

    print("Next steps:")
    print("  1. Run evaluation: python scripts/evaluate_decoder.py")
    print("  2. Check results in: results/decoder_evaluation/")
    print()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
