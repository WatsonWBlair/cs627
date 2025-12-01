"""
Cross-Modal Encoder Alignment Training Script

This script trains Text, Audio, and Video encoders to align in a shared semantic space
using Cross-Modal Momentum Contrastive Learning (MoCo) on the CMU-MOSI dataset.

Usage:
    python src/Training/train_encoder_alignment.py

Features:
    - Momentum encoder with exponential moving average
    - Memory queue for large-scale negative sampling
    - InfoNCE loss for stable contrastive learning
    - Cross-modal alignment (text ↔ audio ↔ video)
    - Periodic checkpointing and validation
    - Cross-modal retrieval evaluation

Based on:
    - Cross-Modal Alignment for End-to-End Spoken Language Understanding (IEEE 2024)
    - Momentum Contrast for Unsupervised Visual Representation Learning (MoCo, CVPR 2020)
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import TrainingArguments
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Encoders.text_2_vec import Text_to_Vec
from Encoders.wav_2_vec import Audio_to_Vec
from Encoders.img_2_vec import Image_to_Vec
from Training.encoder_trainers import Contrast
from Training.Data_Wrangling.mosi_dataset import MOSIDataset, download_mosi, preprocess_mosi, get_mosi_dataloader


# Configuration
DATA_PATH = 'data/cmumosi/'
OUTPUT_DIR = 'results/encoder_alignment/'
ADAPTER_WEIGHTS_DIR = 'AdapterWeights/'

# Training hyperparameters
BATCH_SIZE = 32  # Adjust based on GPU memory (256 is ideal)
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500

# MoCo hyperparameters
MOMENTUM = 0.999
QUEUE_SIZE = 4096  # Start smaller (65536 is ideal but requires more memory)
TEMPERATURE = 0.07
EMBED_DIM = 1024

# Checkpoint and logging
SAVE_STEPS = 1000
LOGGING_STEPS = 100
EVAL_STEPS = 500


def setup_data():
    """
    Download and preprocess CMU-MOSI dataset
    """
    print("=" * 80)
    print("Setting up CMU-MOSI dataset...")
    print("=" * 80)

    # Check if data already exists
    if not os.path.exists(os.path.join(DATA_PATH, 'preprocessed_train.pkl')):
        print("\nData not found. Downloading and preprocessing...")

        # Download dataset
        print("\n[1/2] Downloading CMU-MOSI...")
        download_mosi(DATA_PATH)

        # Preprocess dataset
        print("\n[2/2] Preprocessing CMU-MOSI...")
        preprocess_mosi(DATA_PATH)

        print("\n✓ Dataset ready!")
    else:
        print("\n✓ Preprocessed data found. Skipping download.")


def setup_encoders(device='cuda'):
    """
    Initialize text, audio, and video encoders

    Returns:
        dict: Dictionary of encoders {'text': encoder, 'audio': encoder, 'video': encoder}
    """
    print("\n" + "=" * 80)
    print("Initializing encoders...")
    print("=" * 80)

    # Initialize encoders
    text_encoder = Text_to_Vec().to(device)
    audio_encoder = Audio_to_Vec().to(device)
    video_encoder = Image_to_Vec().to(device)

    # Freeze pretrained models (only train adapters)
    print("\nFreezing pretrained models (only training adapters)...")

    for encoder_name, encoder in [('Text', text_encoder), ('Audio', audio_encoder), ('Video', video_encoder)]:
        # Freeze all parameters
        for param in encoder.parameters():
            param.requires_grad = False

        # Unfreeze adapter
        if hasattr(encoder, 'adapter'):
            for param in encoder.adapter.parameters():
                param.requires_grad = True

            trainable_params = sum(p.numel() for p in encoder.adapter.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in encoder.parameters())

            print(f"  {encoder_name} Encoder: {trainable_params:,} / {total_params:,} trainable params "
                  f"({100 * trainable_params / total_params:.2f}%)")

    encoders = {
        'text': text_encoder,
        'audio': audio_encoder,
        'video': video_encoder
    }

    print("\n✓ Encoders initialized!")
    return encoders


def setup_trainer(encoders, train_dataset):
    """
    Initialize Contrast trainer with MoCo

    Args:
        encoders: Dict of encoder models
        train_dataset: Training dataset

    Returns:
        Contrast: Configured trainer
    """
    print("\n" + "=" * 80)
    print("Initializing MoCo Contrast trainer...")
    print("=" * 80)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type='cosine',
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=5,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )

    trainer = Contrast(
        model=encoders,
        args=training_args,
        train_dataset=train_dataset,
        momentum=MOMENTUM,
        queue_size=QUEUE_SIZE,
        temperature=TEMPERATURE,
        embed_dim=EMBED_DIM,
        use_momentum=True
    )

    print("\n✓ Trainer initialized!")
    return trainer


def evaluate_cross_modal_retrieval(encoders, dataloader, device='cuda', top_k=[1, 5, 10]):
    """
    Evaluate cross-modal retrieval accuracy

    For each text query, rank all audio samples by similarity and check if
    ground-truth audio is in top K.

    Args:
        encoders: Dict of encoder models
        dataloader: Validation DataLoader
        device: Device to use
        top_k: List of K values for Recall@K

    Returns:
        dict: Retrieval metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating cross-modal retrieval...")
    print("=" * 80)

    encoders_eval = {k: v.eval() for k, v in encoders.items()}

    text_embeddings = []
    audio_embeddings = []
    video_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding validation set"):
            # Encode each modality
            if 'text' in batch:
                text_emb = encoders_eval['text'](batch['text'].to(device))
                text_embeddings.append(F.normalize(text_emb, dim=1))

            if 'audio' in batch:
                audio_emb = encoders_eval['audio'](batch['audio'].to(device))
                audio_embeddings.append(F.normalize(audio_emb, dim=1))

            if 'video' in batch:
                video_emb = encoders_eval['video'](batch['video'].to(device))
                video_embeddings.append(F.normalize(video_emb, dim=1))

    # Concatenate all embeddings
    text_embeddings = torch.cat(text_embeddings, dim=0)
    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    video_embeddings = torch.cat(video_embeddings, dim=0)

    # Compute similarity matrices
    text_audio_sim = torch.matmul(text_embeddings, audio_embeddings.T)
    text_video_sim = torch.matmul(text_embeddings, video_embeddings.T)

    # Compute Recall@K for text→audio retrieval
    metrics = {}
    for k in top_k:
        # Text → Audio
        _, top_indices = torch.topk(text_audio_sim, k, dim=1)
        correct_retrievals = (top_indices == torch.arange(len(text_embeddings), device=device).unsqueeze(1)).any(dim=1)
        recall = correct_retrievals.float().mean().item()
        metrics[f'text_to_audio_R@{k}'] = recall

        # Text → Video
        _, top_indices = torch.topk(text_video_sim, k, dim=1)
        correct_retrievals = (top_indices == torch.arange(len(text_embeddings), device=device).unsqueeze(1)).any(dim=1)
        recall = correct_retrievals.float().mean().item()
        metrics[f'text_to_video_R@{k}'] = recall

    # Compute average cosine similarity (semantic alignment quality)
    avg_text_audio_sim = torch.diagonal(text_audio_sim).mean().item()
    avg_text_video_sim = torch.diagonal(text_video_sim).mean().item()

    metrics['avg_text_audio_similarity'] = avg_text_audio_sim
    metrics['avg_text_video_similarity'] = avg_text_video_sim

    print("\nCross-Modal Retrieval Results:")
    print("-" * 80)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 80)

    # Set models back to train mode
    for encoder in encoders.values():
        encoder.train()

    return metrics


def save_adapter_weights(encoders):
    """
    Save adapter weights for all encoders

    Args:
        encoders: Dict of encoder models
    """
    os.makedirs(ADAPTER_WEIGHTS_DIR, exist_ok=True)

    for modality, encoder in encoders.items():
        if hasattr(encoder, 'adapter'):
            encoder.adapter.save()
            print(f"  ✓ Saved {modality} adapter weights")


def main():
    """
    Main training script
    """
    print("\n" + "=" * 80)
    print("Cross-Modal Encoder Alignment Training")
    print("=" * 80)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Step 1: Setup data
    setup_data()

    # Step 2: Load datasets
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)

    train_dataset = MOSIDataset(data_path=DATA_PATH, split='train')
    val_dataset = MOSIDataset(data_path=DATA_PATH, split='valid')

    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 3: Initialize encoders
    encoders = setup_encoders(device)

    # Step 4: Initialize trainer
    trainer = setup_trainer(encoders, train_dataset)

    # Step 5: Initial evaluation
    print("\n" + "=" * 80)
    print("Initial evaluation (before training)...")
    print("=" * 80)
    initial_metrics = evaluate_cross_modal_retrieval(encoders, val_loader, device)

    # Step 6: Training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Step 7: Final evaluation
    print("\n" + "=" * 80)
    print("Final evaluation (after training)...")
    print("=" * 80)
    final_metrics = evaluate_cross_modal_retrieval(encoders, val_loader, device)

    # Step 8: Save adapter weights
    print("\n" + "=" * 80)
    print("Saving adapter weights...")
    print("=" * 80)
    save_adapter_weights(encoders)

    # Summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nMetric Improvements:")
    print("-" * 80)
    for metric in initial_metrics.keys():
        initial = initial_metrics[metric]
        final = final_metrics[metric]
        improvement = final - initial
        print(f"  {metric}:")
        print(f"    Initial: {initial:.4f} → Final: {final:.4f} (Δ {improvement:+.4f})")
    print("-" * 80)

    print(f"\nAdapter weights saved to: {ADAPTER_WEIGHTS_DIR}")
    print(f"Training logs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
