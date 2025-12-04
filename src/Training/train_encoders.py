"""
Cross-Modal Encoder Training using Momentum Contrastive Learning

Trains N encoders jointly to learn a shared semantic vector space from raw multimodal data.
Supports multiple specialized encoders per modality (e.g., tone detector, sarcasm detector for audio).

Based on MoCo (Momentum Contrast) approach with multi-positive InfoNCE loss.

Usage:
    python src/Training/train_encoders.py

Architecture:
    - Encoder Swarm: Multiple encoders per modality, each focusing on specific features
    - Cross-Modal Learning: Each encoder learns from all other encoders as positive pairs
    - Momentum Queues: Separate queue for each encoder to store negative samples
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
import copy
import time
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server compatibility

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec
from src.Encoders.audio.tone_to_vec import Tone_to_Vec
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset
import librosa
from PIL import Image as PILImage

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
MOMENTUM = 0.999
QUEUE_SIZE = 4096  # Increase to 65536 if memory allows
TEMPERATURE = 0.07
SEMANTIC_DIM = 1024


class MultiPositiveInfoNCE(nn.Module):
    """
    InfoNCE loss with multiple positives per query.

    For each encoder's output, all other encoders from the same segment
    are positive pairs, and outputs from other segments are negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        positives: List[torch.Tensor],
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with multiple positives.

        Args:
            query: Query vectors (batch_size, dim)
            positives: List of positive vectors, each (batch_size, dim)
            negatives: Negative vectors from queue (queue_size, dim)

        Returns:
            Loss scalar
        """
        # Normalize all vectors
        query = F.normalize(query, dim=1)
        positives = [F.normalize(pos, dim=1) for pos in positives]
        negatives = F.normalize(negatives, dim=1)

        # Compute similarity with positives
        pos_sims = []
        for pos in positives:
            # (batch_size, batch_size) - diagonal contains correct pairs
            sim = torch.matmul(query, pos.t()) / self.temperature
            pos_sims.append(torch.diagonal(sim))

        # Stack positive similarities (batch_size, num_positives)
        pos_sims = torch.stack(pos_sims, dim=1)

        # Compute similarity with negatives (batch_size, queue_size)
        neg_sims = torch.matmul(query, negatives.t()) / self.temperature

        # Numerator: sum of exp(positive similarities)
        numerator = torch.sum(torch.exp(pos_sims), dim=1)

        # Denominator: numerator + sum of exp(negative similarities)
        denominator = numerator + torch.sum(torch.exp(neg_sims), dim=1)

        # Loss: -log(numerator / denominator)
        loss = -torch.log(numerator / denominator)

        return loss.mean()


class MomentumQueue:
    """
    Queue for storing negative samples with momentum updates.
    """

    def __init__(self, queue_size: int, dim: int, device: str):
        self.queue_size = queue_size
        self.dim = dim
        self.device = device

        # Initialize queue with random vectors
        self.queue = F.normalize(torch.randn(queue_size, dim), dim=1).to(device)
        self.ptr = 0

    @torch.no_grad()
    def update(self, vectors: torch.Tensor):
        """
        Update queue with new vectors.

        Args:
            vectors: New vectors to add (batch_size, dim)
        """
        batch_size = vectors.shape[0]

        # Replace oldest vectors in queue
        if self.ptr + batch_size <= self.queue_size:
            self.queue[self.ptr:self.ptr + batch_size] = vectors
            self.ptr = (self.ptr + batch_size) % self.queue_size
        else:
            # Wrap around
            remaining = self.queue_size - self.ptr
            self.queue[self.ptr:] = vectors[:remaining]
            self.queue[:batch_size - remaining] = vectors[remaining:]
            self.ptr = batch_size - remaining

    def get_queue(self) -> torch.Tensor:
        """Get current queue."""
        return self.queue.clone()


class TrainingReporter:
    """
    Tracks and visualizes training metrics over time.
    """

    def __init__(self, encoder_names: List[str], output_dir: str = "training_reports"):
        """
        Initialize reporter.

        Args:
            encoder_names: List of encoder names to track
            output_dir: Directory to save reports and plots
        """
        self.encoder_names = encoder_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Training metadata
        self.start_time = None
        self.end_time = None
        self.total_epochs = 0

        # Loss history: {encoder_name: [loss per epoch], ...}
        self.loss_history = {name: [] for name in encoder_names}
        self.loss_history['total_loss'] = []

        # Epoch timing
        self.epoch_times = []

    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        print(f"\n[TrainingReporter] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def end_training(self):
        """Mark the end of training."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"\n[TrainingReporter] Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[TrainingReporter] Total training time: {self.format_duration(duration)}")

    def record_epoch(self, epoch: int, metrics: Dict[str, float], epoch_time: float):
        """
        Record metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of losses per encoder
            epoch_time: Time taken for this epoch in seconds
        """
        self.total_epochs = epoch
        self.epoch_times.append(epoch_time)

        # Record losses
        for name in self.encoder_names:
            if name in metrics:
                self.loss_history[name].append(metrics[name])

        if 'total_loss' in metrics:
            self.loss_history['total_loss'].append(metrics['total_loss'])

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

    def save_metrics(self, filename: str = "training_metrics.json"):
        """
        Save all metrics to JSON file.

        Args:
            filename: Output filename
        """
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
            'epoch_times': self.epoch_times,
            'encoder_names': self.encoder_names
        }

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"[TrainingReporter] Saved metrics to {filepath}")

    def create_loss_plots(self):
        """
        Create and save loss visualization plots.
        """
        epochs = list(range(1, self.total_epochs + 1))

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Total loss over time
        ax1 = axes[0]
        ax1.plot(epochs, self.loss_history['total_loss'], 'b-', linewidth=2, label='Total Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss vs Epoch', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Plot 2: Individual encoder losses
        ax2 = axes[1]
        for name in self.encoder_names:
            if name in self.loss_history and len(self.loss_history[name]) > 0:
                ax2.plot(epochs, self.loss_history[name], linewidth=2, label=name, marker='o', markersize=3)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Individual Encoder Losses vs Epoch', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc='best')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.output_dir, 'loss_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[TrainingReporter] Saved loss curves to {plot_path}")

    def create_timing_plot(self):
        """
        Create and save epoch timing plot.
        """
        if not self.epoch_times:
            return

        epochs = list(range(1, len(self.epoch_times) + 1))

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot epoch times
        ax.plot(epochs, self.epoch_times, 'g-', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=np.mean(self.epoch_times), color='r', linestyle='--',
                   linewidth=2, label=f'Average: {np.mean(self.epoch_times):.2f}s')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Epoch Training Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.output_dir, 'epoch_timing.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[TrainingReporter] Saved epoch timing to {plot_path}")

    def print_summary(self):
        """
        Print training summary statistics.
        """
        if not self.start_time or not self.end_time:
            return

        duration = self.end_time - self.start_time

        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)

        print(f"\nTotal Training Time: {self.format_duration(duration)}")
        print(f"Total Epochs: {self.total_epochs}")

        if self.epoch_times:
            avg_epoch_time = np.mean(self.epoch_times)
            min_epoch_time = np.min(self.epoch_times)
            max_epoch_time = np.max(self.epoch_times)
            print(f"Average Epoch Time: {self.format_duration(avg_epoch_time)}")
            print(f"Fastest Epoch: {self.format_duration(min_epoch_time)}")
            print(f"Slowest Epoch: {self.format_duration(max_epoch_time)}")

        print("\nFinal Losses:")
        for name in sorted(self.encoder_names):
            if name in self.loss_history and len(self.loss_history[name]) > 0:
                final_loss = self.loss_history[name][-1]
                initial_loss = self.loss_history[name][0]
                improvement = ((initial_loss - final_loss) / initial_loss) * 100
                print(f"  {name:20s}: {final_loss:.4f} ({improvement:+.1f}% change)")

        if len(self.loss_history['total_loss']) > 0:
            final_total = self.loss_history['total_loss'][-1]
            initial_total = self.loss_history['total_loss'][0]
            improvement = ((initial_total - final_total) / initial_total) * 100
            print(f"  {'Total Loss':20s}: {final_total:.4f} ({improvement:+.1f}% change)")

        print("\nBest Epoch:")
        if len(self.loss_history['total_loss']) > 0:
            best_epoch = np.argmin(self.loss_history['total_loss']) + 1
            best_loss = np.min(self.loss_history['total_loss'])
            print(f"  Epoch {best_epoch} with total loss: {best_loss:.4f}")

        print("=" * 80)


class EncoderConfig:
    """
    Configuration for a single encoder in the swarm.

    Attributes:
        name: Unique identifier for this encoder (e.g., 'text_base', 'audio_tone', 'image_distraction')
        encoder: The encoder module (must have .forward() and .adapter)
        data_key: Key in batch dict to extract input data (e.g., 'text', 'audio', 'video')
        preprocess_fn: Optional function to preprocess data before encoding
        loss_weight: Feature importance weight for loss calculation (default: 1.0)
    """

    def __init__(
        self,
        name: str,
        encoder: nn.Module,
        data_key: str,
        preprocess_fn: Callable[[Any], Any] = None,
        loss_weight: float = 1.0
    ):
        self.name = name
        self.encoder = encoder
        self.data_key = data_key
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.loss_weight = loss_weight

    def __repr__(self):
        return f"EncoderConfig(name='{self.name}', data_key='{self.data_key}')"


class CrossModalTrainer:
    """
    Trainer for cross-modal contrastive learning with N encoders.

    Supports encoder swarms where multiple encoders can process the same modality
    but focus on different features (e.g., tone vs. sarcasm from audio).
    """

    def __init__(
        self,
        encoder_configs: List[EncoderConfig],
        learning_rate: float = 1e-4,
        momentum: float = 0.999,
        queue_size: int = 4096,
        temperature: float = 0.07,
        device: str = DEVICE
    ):
        """
        Initialize trainer with N encoders.

        Args:
            encoder_configs: List of EncoderConfig objects
            learning_rate: Learning rate for optimizer
            momentum: Momentum coefficient for momentum encoders
            queue_size: Size of negative sample queue per encoder
            temperature: Temperature for contrastive loss
            device: Device to run on
        """
        self.device = device
        self.momentum = momentum
        self.encoder_configs = encoder_configs
        self.num_encoders = len(encoder_configs)

        print(f"\n[CrossModalTrainer] Initializing with {self.num_encoders} encoders:")
        for config in encoder_configs:
            print(f"  - {config.name} (data_key: {config.data_key})")

        # Move encoders to device
        self.encoders = {}
        self.encoders_m = {}  # Momentum encoders
        self.queues = {}

        for config in encoder_configs:
            # Query encoder
            encoder = config.encoder.to(device)
            self.encoders[config.name] = encoder

            # Momentum encoder (copy)
            encoder_m = copy.deepcopy(encoder).to(device)
            for param in encoder_m.parameters():
                param.requires_grad = False
            self.encoders_m[config.name] = encoder_m

            # Freeze pretrained model, only train adapter
            if hasattr(encoder, 'encoder'):
                for param in encoder.encoder.parameters():
                    param.requires_grad = False

            # Create queue for this encoder
            self.queues[config.name] = MomentumQueue(queue_size, SEMANTIC_DIM, device)

        # Create optimizer for all adapters
        adapter_params = []
        for config in encoder_configs:
            if hasattr(config.encoder, 'adapter'):
                adapter_params.append({'params': config.encoder.adapter.parameters()})

        self.optimizer = torch.optim.Adam(adapter_params, lr=learning_rate)

        # Create loss function
        self.criterion = MultiPositiveInfoNCE(temperature=temperature)

        print(f"[CrossModalTrainer] Optimizer created with {len(adapter_params)} adapter parameter groups")

    @torch.no_grad()
    def update_momentum_encoders(self):
        """
        Update all momentum encoders using exponential moving average.
        """
        for config in self.encoder_configs:
            encoder_q = self.encoders[config.name]
            encoder_k = self.encoders_m[config.name]

            for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def encode_batch(
        self,
        batch: Dict,
        use_momentum: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batch with all encoders.

        Args:
            batch: Dictionary with data for all modalities
            use_momentum: Whether to use momentum encoders

        Returns:
            Dictionary mapping encoder names to output vectors
        """
        outputs = {}
        encoders = self.encoders_m if use_momentum else self.encoders

        for config in self.encoder_configs:
            # Get data for this encoder
            data = batch[config.data_key]

            # Preprocess if needed
            data = config.preprocess_fn(data)

            # Encode
            encoder = encoders[config.name]
            vec = encoder(data, device=self.device)

            outputs[config.name] = vec

        return outputs

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step on a batch.

        Args:
            batch: Dictionary with data for all modalities

        Returns:
            Dictionary mapping encoder names to their losses
        """
        # Encode with query encoders
        query_outputs = self.encode_batch(batch, use_momentum=False)

        # Encode with momentum encoders (no gradients)
        with torch.no_grad():
            key_outputs = self.encode_batch(batch, use_momentum=True)

        # Compute loss for each encoder
        losses = {}

        for config in self.encoder_configs:
            query_vec = query_outputs[config.name]

            # Positives: all OTHER encoders from same segment
            positive_vecs = [
                key_outputs[other_config.name]
                for other_config in self.encoder_configs
                if other_config.name != config.name
            ]

            # Negatives: this encoder's queue
            negative_vecs = self.queues[config.name].get_queue()

            # Compute loss
            loss = self.criterion(
                query=query_vec,
                positives=positive_vecs,
                negatives=negative_vecs
            )

            losses[config.name] = loss

        # Total loss (weighted sum of all encoder losses)
        total_loss = sum(
            config.loss_weight * losses[config.name]
            for config in self.encoder_configs
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update momentum encoders
        self.update_momentum_encoders()

        # Update queues with momentum encoder outputs
        for config in self.encoder_configs:
            self.queues[config.name].update(key_outputs[config.name])

        # Return losses as scalars
        return {name: loss.item() for name, loss in losses.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Dictionary with average losses per encoder
        """
        # Set all encoders to train mode
        for encoder in self.encoders.values():
            encoder.train()

        # Accumulate losses
        total_losses = {config.name: 0.0 for config in self.encoder_configs}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Train step
            step_losses = self.train_step(batch)

            # Accumulate
            for name, loss in step_losses.items():
                total_losses[name] += loss
            num_batches += 1

            # Update progress bar
            total_step_loss = sum(step_losses.values())
            postfix = {'total_loss': f'{total_step_loss:.4f}'}
            for name, loss in step_losses.items():
                postfix[name] = f'{loss:.4f}'
            pbar.set_postfix(postfix)

        # Average losses
        avg_losses = {
            name: total / num_batches
            for name, total in total_losses.items()
        }
        avg_losses['total_loss'] = sum(avg_losses.values())

        return avg_losses

    def save_adapters(self):
        """
        Save all adapter weights.
        """
        for config in self.encoder_configs:
            encoder = self.encoders[config.name]
            if hasattr(encoder, 'adapter'):
                encoder.adapter.save()
                print(f"[OK] Saved {config.name} adapter: {encoder.adapter.weights_path}")


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle variable-length sequences.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched data
    """
    # Extract text (list of strings)
    texts = [sample['text'] for sample in batch]

    # Extract audio features (pad to max length in batch)
    audio_features = [torch.tensor(sample['audio'], dtype=torch.float32) for sample in batch]
    # Get max sequence length
    max_audio_len = max([a.shape[0] for a in audio_features])
    # Pad to max length
    audio_padded = []
    for audio in audio_features:
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        pad_len = max_audio_len - audio.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, audio.shape[1])
            audio = torch.cat([audio, padding], dim=0)
        audio_padded.append(audio)
    audio_batch = torch.stack(audio_padded)

    # Extract video features (pad to max length in batch)
    video_features = [torch.tensor(sample['video'], dtype=torch.float32) for sample in batch]
    max_video_len = max([v.shape[0] for v in video_features])
    # Pad to max length
    video_padded = []
    for video in video_features:
        if len(video.shape) == 1:
            video = video.unsqueeze(0)
        pad_len = max_video_len - video.shape[0]
        if pad_len > 0:
            padding = torch.zeros(pad_len, video.shape[1])
            video = torch.cat([video, padding], dim=0)
        video_padded.append(video)
    video_batch = torch.stack(video_padded)

    return {
        'text': texts,
        'audio': audio_batch,
        'video': video_batch
    }


def collate_fn_raw_video(batch: List[Dict]) -> Dict:
    """
    Collate function for eagerly-loaded raw video/audio data.

    Data is already loaded into RAM by the dataset. This function just
    extracts and converts to the format expected by encoders.

    Args:
        batch: List of samples from MOSIRawVideoDataset

    Returns:
        Dictionary with batched data:
            - 'text': List of raw text strings
            - 'audio': List of audio waveforms (numpy arrays, variable length)
            - 'video': List of PIL Images (converted from numpy arrays)
    """
    # Extract text (raw strings)
    texts = [sample['text'] for sample in batch]

    # Extract audio waveforms (already loaded as numpy arrays)
    audio_waveforms = []
    for sample in batch:
        if sample['audio'] is not None:
            audio_waveforms.append(sample['audio'])
        else:
            # Use zero waveform as fallback if modality is missing
            audio_waveforms.append(np.zeros(16000, dtype=np.float32))  # 1 second of silence

    # Convert video frames from numpy to PIL Images (for ViT encoder)
    video_frames = []
    for sample in batch:
        if sample['video'] is not None:
            # Convert numpy array (H, W, 3) to PIL Image
            frame = PILImage.fromarray(sample['video'])
            video_frames.append(frame)
        else:
            # Use blank frame as fallback if modality is missing
            video_frames.append(PILImage.new('RGB', (224, 224), color='black'))

    return {
        'text': texts,
        'audio': audio_waveforms,
        'video': video_frames
    }


def main():
    """Main training function."""
    print("=" * 80)
    print("Cross-Modal Encoder Training (N Encoder Support)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Momentum: {MOMENTUM}")
    print(f"Queue size: {QUEUE_SIZE}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 80)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    try:
        # Use MOSIRawVideoDataset for TRUE raw multimodal data from extracted videos
        # This bypasses preprocessed features and loads raw audio waveforms + video frames
        train_dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path='data/cmumosi/mosi/',
            audio_dir='data/cmumosi/audio/',
            video_dir='data/cmumosi/frames/',
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )
        print(f"[OK] Loaded {len(train_dataset)} training samples")

        if len(train_dataset) == 0:
            print("[ERROR] Dataset is empty! Have you extracted audio and frames?")
            print("Run: python extract_test_segments.py")
            return

    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create dataloader with lazy loading collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_raw_video,  # Use lazy loading collate function
        num_workers=0  # Set to 0 for Windows compatibility
    )

    # Initialize encoders
    print("\n[2/5] Initializing encoder swarm...")

    # Define encoder configurations
    # This is where you specify N encoders with their specializations
    encoder_configs = [
        # Text encoder - baseline weight
        EncoderConfig(
            name='text_base',
            encoder=Text_to_Vec(),
            data_key='text',
            loss_weight=1.0  # Baseline (can increase for conversational coherence)
        ),

        # Audio encoder #1: Waveform/content (Whisper)
        EncoderConfig(
            name='audio_waveform',
            encoder=Audio_to_Vec(),
            data_key='audio',
            loss_weight=1.0  # Equal to baseline
        ),

        # Audio encoder #2: Tone/prosody (WavLM)
        EncoderConfig(
            name='audio_tone',
            encoder=Tone_to_Vec(),  # Uses WavLM-base for tone detection
            data_key='audio',       # Same raw waveform input!
            loss_weight=1.0         # Equal to baseline initially
        ),

        # Image encoder - baseline weight
        EncoderConfig(
            name='image_base',
            encoder=Image_to_Vec(),
            data_key='video',
            loss_weight=1.0  # Equal to baseline
        ),

        # Future specialized encoders (examples):
        # EncoderConfig(
        #     name='audio_sarcasm',
        #     encoder=Sarcasm_to_Vec(),  # Separate model for sarcasm detection
        #     data_key='audio',           # Same audio input as above!
        #     loss_weight=1.0
        # ),
        # EncoderConfig(
        #     name='image_distraction',
        #     encoder=Distraction_to_Vec(),  # Attention/distraction from video
        #     data_key='video',               # Same video input
        #     loss_weight=1.0
        # ),
    ]

    print(f"[OK] Initialized {len(encoder_configs)} encoders")

    # Create trainer
    print("\n[3/5] Creating trainer...")
    trainer = CrossModalTrainer(
        encoder_configs=encoder_configs,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        queue_size=QUEUE_SIZE,
        temperature=TEMPERATURE,
        device=DEVICE
    )
    print("[OK] Trainer created")

    # Create reporter
    print("\n[3.5/5] Creating training reporter...")
    encoder_names = [config.name for config in encoder_configs]
    reporter = TrainingReporter(encoder_names=encoder_names, output_dir="training_reports")
    print("[OK] Reporter created")

    # Start training
    print("\n[4/5] Starting training...")
    reporter.start_training()
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'=' * 80}")

        # Track epoch time
        epoch_start_time = time.time()

        # Train for one epoch
        metrics = trainer.train_epoch(train_loader, epoch)

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        # Record metrics
        reporter.record_epoch(epoch, metrics, epoch_duration)

        # Print metrics
        print(f"\nEpoch {epoch} Results:")
        for name in sorted(metrics.keys()):
            if name != 'total_loss':
                print(f"  {name:20s}: {metrics[name]:.4f}")
        print(f"  {'total_loss':20s}: {metrics['total_loss']:.4f}")
        print(f"  Epoch Time: {reporter.format_duration(epoch_duration)}")

        # Save adapters if best loss
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            print(f"\n[OK] New best loss: {best_loss:.4f}")
            trainer.save_adapters()

        # Save every 10 epochs
        if epoch % 10 == 0:
            print(f"\n[OK] Checkpoint at epoch {epoch}")
            trainer.save_adapters()

    # End training
    reporter.end_training()

    # Final save
    print("\n[5/5] Training complete!")
    print("Saving final adapter weights...")
    trainer.save_adapters()

    # Generate reports
    print("\n[6/6] Generating training reports...")
    reporter.save_metrics()
    reporter.create_loss_plots()
    reporter.create_timing_plot()
    reporter.print_summary()

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nAdapter weights saved for:")
    for config in encoder_configs:
        if hasattr(config.encoder, 'adapter'):
            print(f"  - {config.name}: {config.encoder.adapter.weights_path}")
    print("\nTraining reports saved to: training_reports/")
    print("  - training_metrics.json (detailed metrics)")
    print("  - loss_curves.png (loss visualization)")
    print("  - epoch_timing.png (timing analysis)")
    print("\nNext steps:")
    print("  1. Review training reports in training_reports/")
    print("  2. Validate encoders produce aligned vectors")
    print("  3. Train text decoder (python src/Training/train_text_decoder.py)")
    print("  4. Add specialized encoders for tone, sarcasm, distraction, etc.")
    print("  5. Test end-to-end pipeline")


if __name__ == "__main__":
    main()
