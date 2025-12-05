"""
Unified Cross-Modal Decoder Training Infrastructure

Mirrors the encoder training architecture (CrossModalTrainer) for decoder training.
Supports simultaneous training of multiple decoders (text, audio, image) with:
- Configurable semantic vector sources (text_only, matched, mixed)
- Modality-specific loss functions
- Two-stage training for non-differentiable decoders (audio)
- Semantic fidelity monitoring (not in loss function)

Usage:
    trainer = CrossModalDecoderTrainer(
        decoder_configs=[...],
        source_encoders={...},
        reference_encoders={...},
        semantic_source='mixed'
    )
    metrics = trainer.train_epoch(dataloader)

Based on: src/Training/encoder_trainers.py (CrossModalTrainer)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
import copy
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import random

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default dimensions
SEMANTIC_DIM = 1024


@dataclass
class DecoderConfig:
    """
    Configuration for a single decoder in the training pipeline.

    Mirrors EncoderConfig from encoder_trainers.py but for decoders.

    Attributes:
        name: Unique identifier (e.g., 'text_base', 'audio_waveform')
        decoder: The decoder module (must have .adapter attribute)
        target_modality: Output modality ('text', 'audio', 'image')
        loss_fn: Modality-specific loss function
        loss_weight: Weight in combined loss (default: 1.0)
        training_stage: Stage for two-stage training ('proxy', 'generation', 'full')
        source_encoder_key: Which encoder to use for semantic vectors (None = use semantic_source)
    """
    name: str
    decoder: nn.Module
    target_modality: str
    loss_fn: Callable
    loss_weight: float = 1.0
    training_stage: str = 'full'  # 'proxy', 'generation', 'full'
    source_encoder_key: Optional[str] = None  # Override semantic_source for this decoder

    def __repr__(self):
        return f"DecoderConfig(name='{self.name}', target_modality='{self.target_modality}', stage='{self.training_stage}')"


class DecoderTrainingReporter:
    """
    Tracks and visualizes decoder training metrics over time.

    Based on TrainingReporter from train_encoders.py but adapted for decoder metrics.
    """

    def __init__(self, decoder_names: List[str], output_dir: str = "training_reports/decoders"):
        """
        Initialize reporter.

        Args:
            decoder_names: List of decoder names to track
            output_dir: Directory to save reports and plots
        """
        self.decoder_names = decoder_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Training metadata
        self.start_time = None
        self.end_time = None
        self.total_epochs = 0

        # Loss history per decoder
        self.loss_history = {name: [] for name in decoder_names}
        self.loss_history['total_loss'] = []

        # Semantic fidelity history (monitoring only)
        self.semantic_fidelity_history = {name: [] for name in decoder_names}

        # Epoch timing
        self.epoch_times = []

    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        print(f"\n[DecoderTrainingReporter] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def end_training(self):
        """Mark the end of training."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"\n[DecoderTrainingReporter] Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[DecoderTrainingReporter] Total training time: {self.format_duration(duration)}")

    def record_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        semantic_metrics: Dict[str, float],
        epoch_time: float
    ):
        """
        Record metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of losses per decoder
            semantic_metrics: Dictionary of semantic fidelity per decoder
            epoch_time: Time taken for this epoch in seconds
        """
        self.total_epochs = epoch
        self.epoch_times.append(epoch_time)

        # Record losses
        for name in self.decoder_names:
            if name in metrics:
                self.loss_history[name].append(metrics[name])

        if 'total_loss' in metrics:
            self.loss_history['total_loss'].append(metrics['total_loss'])

        # Record semantic fidelity
        for name in self.decoder_names:
            if name in semantic_metrics:
                self.semantic_fidelity_history[name].append(semantic_metrics[name])

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        delta = timedelta(seconds=int(seconds))
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if delta.days > 0:
            return f"{delta.days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def save_metrics(self, filename: str = "decoder_training_metrics.json"):
        """Save all metrics to JSON file."""
        filepath = os.path.join(self.output_dir, filename)

        metrics = {
            'training_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'total_duration_seconds': self.end_time - self.start_time if self.end_time and self.start_time else None,
                'total_epochs': self.total_epochs,
                'avg_epoch_time_seconds': np.mean(self.epoch_times) if self.epoch_times else None,
            },
            'loss_history': self.loss_history,
            'semantic_fidelity_history': self.semantic_fidelity_history,
            'epoch_times': self.epoch_times,
            'decoder_names': self.decoder_names
        }

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"[DecoderTrainingReporter] Saved metrics to {filepath}")

    def print_summary(self):
        """Print training summary statistics."""
        if not self.start_time or not self.end_time:
            return

        duration = self.end_time - self.start_time

        print("\n" + "=" * 80)
        print("DECODER TRAINING SUMMARY")
        print("=" * 80)

        print(f"\nTotal Training Time: {self.format_duration(duration)}")
        print(f"Total Epochs: {self.total_epochs}")

        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            print(f"Average Epoch Time: {self.format_duration(avg_epoch_time)}")

        print("\nFinal Losses:")
        for name in sorted(self.decoder_names):
            if name in self.loss_history and len(self.loss_history[name]) > 0:
                final_loss = self.loss_history[name][-1]
                initial_loss = self.loss_history[name][0]
                if initial_loss > 0:
                    improvement = ((initial_loss - final_loss) / initial_loss) * 100
                    print(f"  {name:20s}: {final_loss:.4f} ({improvement:+.1f}% change)")
                else:
                    print(f"  {name:20s}: {final_loss:.4f}")

        print("\nFinal Semantic Fidelity:")
        for name in sorted(self.decoder_names):
            if name in self.semantic_fidelity_history and len(self.semantic_fidelity_history[name]) > 0:
                final_fidelity = self.semantic_fidelity_history[name][-1]
                print(f"  {name:20s}: {final_fidelity:.4f}")

        print("=" * 80)


class CrossModalDecoderTrainer:
    """
    Unified trainer for cross-modal decoder learning with N decoders.

    Mirrors CrossModalTrainer from encoder_trainers.py but for decoders.

    Features:
    - Simultaneous training of multiple decoders
    - Configurable semantic vector sources (text_only, matched, mixed)
    - Two-stage training support for non-differentiable decoders
    - Semantic fidelity monitoring (separate from loss)
    - Single optimizer for all decoder adapters
    """

    def __init__(
        self,
        decoder_configs: List[DecoderConfig],
        source_encoders: Dict[str, nn.Module],
        reference_encoders: Optional[Dict[str, nn.Module]] = None,
        semantic_source: str = 'text_only',  # 'text_only', 'matched', 'mixed'
        learning_rate: float = 1e-4,
        device: str = DEVICE
    ):
        """
        Initialize trainer with N decoders.

        Args:
            decoder_configs: List of DecoderConfig objects
            source_encoders: Dict of encoders for generating semantic vectors
                            e.g., {'text': Text_to_Vec(), 'audio': Audio_to_Vec()}
            reference_encoders: Dict of encoders for semantic fidelity monitoring
                               (defaults to source_encoders if None)
            semantic_source: How to generate semantic vectors:
                           - 'text_only': Use text encoder for all decoders
                           - 'matched': Each decoder uses its modality's encoder
                           - 'mixed': Random sampling from all available encoders
            learning_rate: Learning rate for optimizer
            device: Device to run on
        """
        self.device = device
        self.decoder_configs = decoder_configs
        self.num_decoders = len(decoder_configs)
        self.semantic_source = semantic_source

        print(f"\n[CrossModalDecoderTrainer] Initializing with {self.num_decoders} decoders:")
        for config in decoder_configs:
            print(f"  - {config.name} (target: {config.target_modality}, stage: {config.training_stage})")
        print(f"  Semantic source: {semantic_source}")

        # Store encoders (frozen, used for semantic vector generation)
        self.source_encoders = {}
        for name, encoder in source_encoders.items():
            encoder = encoder.to(device)
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
            self.source_encoders[name] = encoder

        print(f"[CrossModalDecoderTrainer] Source encoders: {list(self.source_encoders.keys())}")

        # Reference encoders for semantic fidelity (defaults to source encoders)
        self.reference_encoders = reference_encoders if reference_encoders else source_encoders
        for name, encoder in self.reference_encoders.items():
            encoder = encoder.to(device)
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

        # Move decoders to device
        self.decoders = {}
        for config in decoder_configs:
            decoder = config.decoder.to(device)
            self.decoders[config.name] = decoder

            # Freeze pretrained model, only train adapter
            if hasattr(decoder, 'decoder'):
                for param in decoder.decoder.parameters():
                    param.requires_grad = False
            # For audio decoder, freeze TTS model
            if hasattr(decoder, 'tts_model'):
                for param in decoder.tts_model.parameters():
                    param.requires_grad = False
            if hasattr(decoder, 'vocoder'):
                for param in decoder.vocoder.parameters():
                    param.requires_grad = False

        # Create optimizer for all adapter parameters
        adapter_params = []
        for config in decoder_configs:
            decoder = self.decoders[config.name]
            if hasattr(decoder, 'adapter'):
                adapter_params.append({'params': decoder.adapter.parameters()})

        self.optimizer = torch.optim.Adam(adapter_params, lr=learning_rate)

        print(f"[CrossModalDecoderTrainer] Optimizer created with {len(adapter_params)} adapter parameter groups")

    def get_semantic_vectors(
        self,
        batch: Dict,
        decoder_config: Optional[DecoderConfig] = None
    ) -> torch.Tensor:
        """
        Generate semantic vectors from batch using configured source.

        Args:
            batch: Dictionary with data for all modalities
            decoder_config: Optional specific decoder config (for 'matched' mode)

        Returns:
            Semantic vectors tensor (batch_size, SEMANTIC_DIM)
        """
        with torch.no_grad():
            if self.semantic_source == 'text_only':
                # Always use text encoder
                if 'text' in self.source_encoders and 'text' in batch:
                    return self.source_encoders['text'](batch['text'], device=self.device)
                else:
                    raise ValueError("text_only mode requires 'text' encoder and data")

            elif self.semantic_source == 'matched':
                # Use encoder matching decoder's target modality
                if decoder_config is None:
                    raise ValueError("matched mode requires decoder_config")

                # Map target modality to encoder key
                modality_map = {
                    'text': 'text',
                    'audio': 'audio',
                    'image': 'image'
                }
                encoder_key = modality_map.get(decoder_config.target_modality, 'text')

                # Override if decoder has specific source encoder
                if decoder_config.source_encoder_key:
                    encoder_key = decoder_config.source_encoder_key

                if encoder_key in self.source_encoders:
                    data_key = encoder_key if encoder_key in batch else 'text'
                    return self.source_encoders[encoder_key](batch[data_key], device=self.device)
                else:
                    # Fallback to text
                    return self.source_encoders['text'](batch['text'], device=self.device)

            elif self.semantic_source == 'mixed':
                # Random sampling from available encoders
                available_encoders = []
                for enc_name, encoder in self.source_encoders.items():
                    # Check if we have data for this encoder
                    data_key = enc_name if enc_name in batch else None
                    if data_key and batch.get(data_key) is not None:
                        available_encoders.append((enc_name, data_key))

                if not available_encoders:
                    raise ValueError("No available encoders match batch data")

                # Random selection
                enc_name, data_key = random.choice(available_encoders)
                return self.source_encoders[enc_name](batch[data_key], device=self.device)

            else:
                raise ValueError(f"Unknown semantic_source: {self.semantic_source}")

    def compute_losses(
        self,
        batch: Dict,
        semantic_vectors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all decoders.

        Args:
            batch: Dictionary with target data for all modalities
            semantic_vectors: Dictionary mapping decoder names to semantic vectors

        Returns:
            Dictionary mapping decoder names to their losses
        """
        losses = {}

        for config in self.decoder_configs:
            decoder = self.decoders[config.name]
            sem_vec = semantic_vectors[config.name]

            # Get target data based on modality
            if config.target_modality == 'text':
                target = batch.get('text')
            elif config.target_modality == 'audio':
                target = batch.get('audio')
            elif config.target_modality == 'image':
                target = batch.get('video')  # 'video' key contains frames
            else:
                target = batch.get(config.target_modality)

            if target is None:
                print(f"[WARNING] No target data for {config.name}")
                continue

            # Compute loss using config's loss function
            try:
                loss = config.loss_fn(
                    semantic_vectors=sem_vec,
                    target=target,
                    decoder=decoder,
                    device=self.device,
                    training_stage=config.training_stage
                )
                losses[config.name] = loss
            except Exception as e:
                print(f"[ERROR] Loss computation failed for {config.name}: {e}")
                continue

        return losses

    @torch.no_grad()
    def evaluate_semantic_fidelity(
        self,
        batch: Dict,
        semantic_vectors: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate semantic fidelity for all decoders (monitoring only).

        This measures semantic drift: decode -> re-encode -> compare to original.
        NOT used in loss function, only for monitoring.

        Args:
            batch: Dictionary with original data
            semantic_vectors: Dictionary mapping decoder names to semantic vectors

        Returns:
            Dictionary mapping decoder names to semantic fidelity scores
        """
        fidelity_scores = {}

        for config in self.decoder_configs:
            decoder = self.decoders[config.name]
            original_vec = semantic_vectors[config.name]

            try:
                # Generate output from semantic vector
                if config.target_modality == 'text':
                    # Text decoder: generate text, re-encode
                    generated = decoder(original_vec, device=self.device)
                    if isinstance(generated, str):
                        generated = [generated]

                    if 'text' in self.reference_encoders:
                        reconstructed_vec = self.reference_encoders['text'](
                            generated, device=self.device
                        )
                    else:
                        continue

                elif config.target_modality == 'audio':
                    # Audio decoder: generate audio, re-encode
                    # This is expensive, so we might skip during training
                    continue  # Skip audio fidelity for now (expensive)

                elif config.target_modality == 'image':
                    # Image decoder: skip for now (expensive)
                    continue

                else:
                    continue

                # Compute cosine similarity
                original_norm = F.normalize(original_vec, dim=1)
                reconstructed_norm = F.normalize(reconstructed_vec, dim=1)
                cosine_sim = F.cosine_similarity(original_norm, reconstructed_norm, dim=1)

                fidelity_scores[config.name] = cosine_sim.mean().item()

            except Exception as e:
                print(f"[WARNING] Semantic fidelity evaluation failed for {config.name}: {e}")
                continue

        return fidelity_scores

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step on a batch.

        Args:
            batch: Dictionary with data for all modalities

        Returns:
            Dictionary mapping decoder names to their losses
        """
        # Set decoders to train mode (adapters)
        for config in self.decoder_configs:
            if hasattr(self.decoders[config.name], 'adapter'):
                self.decoders[config.name].adapter.train()

        # Generate semantic vectors for each decoder
        semantic_vectors = {}
        for config in self.decoder_configs:
            sem_vec = self.get_semantic_vectors(batch, config)
            semantic_vectors[config.name] = sem_vec

        # Compute losses
        losses = self.compute_losses(batch, semantic_vectors)

        if not losses:
            return {}

        # Total loss (weighted sum)
        total_loss = sum(
            config.loss_weight * losses.get(config.name, torch.tensor(0.0))
            for config in self.decoder_configs
            if config.name in losses
        )

        # Backward pass
        self.optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            self.optimizer.step()

        # Return losses as scalars
        return {name: loss.item() if isinstance(loss, torch.Tensor) else loss
                for name, loss in losses.items()}

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        eval_fidelity_every: int = 50
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            eval_fidelity_every: Evaluate semantic fidelity every N batches (0 = only at end)

        Returns:
            Tuple of (average_losses, average_fidelity_scores)
        """
        # Accumulate losses
        total_losses = {config.name: 0.0 for config in self.decoder_configs}
        total_fidelity = {config.name: 0.0 for config in self.decoder_configs}
        num_batches = 0
        num_fidelity_evals = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar, 1):
            # Training step
            step_losses = self.train_step(batch)

            # Accumulate losses
            for name, loss in step_losses.items():
                total_losses[name] += loss
            num_batches += 1

            # Periodic semantic fidelity evaluation
            should_eval = eval_fidelity_every > 0 and batch_idx % eval_fidelity_every == 0
            if should_eval:
                # Generate semantic vectors for evaluation
                semantic_vectors = {}
                for config in self.decoder_configs:
                    with torch.no_grad():
                        sem_vec = self.get_semantic_vectors(batch, config)
                        semantic_vectors[config.name] = sem_vec

                fidelity_scores = self.evaluate_semantic_fidelity(batch, semantic_vectors)
                for name, score in fidelity_scores.items():
                    total_fidelity[name] += score
                num_fidelity_evals += 1

            # Update progress bar
            total_step_loss = sum(step_losses.values())
            postfix = {'total_loss': f'{total_step_loss:.4f}'}
            for name, loss in step_losses.items():
                postfix[name[:10]] = f'{loss:.4f}'
            pbar.set_postfix(postfix)

        # Average losses
        avg_losses = {
            name: total / num_batches if num_batches > 0 else 0.0
            for name, total in total_losses.items()
        }
        avg_losses['total_loss'] = sum(avg_losses.values())

        # Average fidelity scores
        avg_fidelity = {
            name: total / num_fidelity_evals if num_fidelity_evals > 0 else 0.0
            for name, total in total_fidelity.items()
        }

        return avg_losses, avg_fidelity

    def save_all_adapters(self):
        """Save all decoder adapter weights."""
        for config in self.decoder_configs:
            decoder = self.decoders[config.name]
            if hasattr(decoder, 'adapter'):
                decoder.adapter.save()
                print(f"[OK] Saved {config.name} adapter: {decoder.adapter.weights_path}")

    def update_training_stage(self, decoder_name: str, new_stage: str):
        """
        Update training stage for a specific decoder.

        Used for two-stage audio training: 'proxy' -> 'generation'

        Args:
            decoder_name: Name of decoder to update
            new_stage: New training stage ('proxy', 'generation', 'full')
        """
        for config in self.decoder_configs:
            if config.name == decoder_name:
                old_stage = config.training_stage
                config.training_stage = new_stage
                print(f"[CrossModalDecoderTrainer] {decoder_name}: stage {old_stage} -> {new_stage}")
                return

        print(f"[WARNING] Decoder {decoder_name} not found")
