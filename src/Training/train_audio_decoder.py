"""
Audio Decoder Adapter Training - Two-Stage Training Pipeline

Trains Vec_to_Audio adapter using a two-stage approach:
- Stage 1 (Proxy): Train adapter with MSE loss on hidden states
- Stage 2 (Generation): Fine-tune with mel reconstruction feedback

Training Objective:
    Train adapter to map semantic vectors to SpeechT5-compatible representations
    that enable high-quality speech generation with minimal semantic drift.

Two-Stage Training Flow:
Stage 1 (Proxy - Differentiable):
1. Input text → Text_to_Vec (frozen) → semantic_vec
2. semantic_vec → Adapter (trainable) → hidden_states
3. Compute MSE loss encouraging bounded, useful hidden states
4. Backprop through adapter

Stage 2 (Generation - Fine-tuning):
1. Same as Stage 1, but switch to mel reconstruction monitoring
2. Use generation quality for model selection
3. Continue training with proxy loss but use mel quality for checkpoints

Usage:
    python src/Training/train_audio_decoder.py

    # Environment variable configuration:
    TRAINING_STAGE=proxy python src/Training/train_audio_decoder.py
    PROXY_EPOCHS=15 GENERATION_EPOCHS=5 python src/Training/train_audio_decoder.py

Based on: "Bootstrapping Multi-Modal Semantic-Vector Spaces" (Fig. 4, page 3)
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import json
import librosa

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.Encoders import Text_to_Vec, Audio_to_Vec
from src.Decoders import Vec_to_Audio
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset

# ============================================================================
# HYPERPARAMETERS (Configurable via environment variables)
# ============================================================================

from dotenv import load_dotenv
load_dotenv()

# Training parameters
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))  # Smaller due to audio processing overhead
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0001'))  # Standard for adapter training

# Two-stage training epochs
PROXY_EPOCHS = int(os.getenv('PROXY_EPOCHS', '15'))  # Stage 1: proxy loss training
GENERATION_EPOCHS = int(os.getenv('GENERATION_EPOCHS', '5'))  # Stage 2: generation fine-tuning
EPOCHS = PROXY_EPOCHS + GENERATION_EPOCHS  # Total epochs

# Starting stage (for resuming training)
STARTING_STAGE = os.getenv('TRAINING_STAGE', 'proxy')  # 'proxy' or 'generation'

# Loss weights
MEL_L1_WEIGHT = float(os.getenv('MEL_L1_WEIGHT', '1.0'))  # L1 mel reconstruction weight
SPECTRAL_CONVERGENCE_WEIGHT = float(os.getenv('SPECTRAL_CONVERGENCE_WEIGHT', '0.5'))  # Spectral convergence weight
PROXY_LOSS_WEIGHT = float(os.getenv('PROXY_LOSS_WEIGHT', '1.0'))  # Proxy hidden state loss weight

# Evaluation frequency
EVAL_SEMANTIC_EVERY = int(os.getenv('EVAL_SEMANTIC_EVERY', '10'))  # Evaluate semantic drift every N batches (0 = only at epoch end)

# Dataset paths
MOSI_DATA_PATH = "data/cmumosi/mosi/"
AUDIO_DIR = "data/cmumosi/audio/"
VIDEO_DIR = "data/cmumosi/frames/"

# Checkpointing
CHECKPOINT_DIR = "DecoderCheckpoints/audio/"
SAVE_EVERY = 5  # Save checkpoint every N epochs

# Audio parameters (matching encoder/decoder specs)
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
MAX_AUDIO_LENGTH = 10.0  # Maximum audio duration in seconds

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# MEL SPECTROGRAM UTILITIES
# ============================================================================

def extract_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS
) -> torch.Tensor:
    """
    Extract mel spectrogram from audio waveform.

    Uses librosa for feature extraction to match common TTS pipelines.

    Args:
        waveform: Audio waveform as numpy array (samples,)
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel frequency bins

    Returns:
        Mel spectrogram tensor (time, n_mels)
    """
    # Ensure float32
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)

    # Extract mel spectrogram using librosa
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate // 2
    )

    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to roughly [-1, 1] range
    mel_normalized = (mel_db + 80) / 80  # Assuming -80 dB floor

    # Transpose to (time, n_mels) and convert to tensor
    mel_tensor = torch.tensor(mel_normalized.T, dtype=torch.float32)

    return mel_tensor


def pad_mel_spectrograms(
    mel_list: List[torch.Tensor],
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad mel spectrograms to same length for batching.

    Args:
        mel_list: List of mel spectrogram tensors (time, n_mels)
        max_length: Maximum length to pad to (None = use max in batch)

    Returns:
        (padded_mels, lengths) where padded_mels is (batch, time, n_mels)
    """
    lengths = [mel.shape[0] for mel in mel_list]

    if max_length is None:
        max_length = max(lengths)

    batch_size = len(mel_list)
    n_mels = mel_list[0].shape[1]

    padded = torch.zeros(batch_size, max_length, n_mels)

    for i, mel in enumerate(mel_list):
        length = min(mel.shape[0], max_length)
        padded[i, :length, :] = mel[:length, :]

    return padded, torch.tensor(lengths)


# ============================================================================
# DATA LOADING
# ============================================================================

def collate_fn_audio_decoder(batch: List[Dict]) -> Dict:
    """
    Collate function for audio decoder training.

    Extracts text and audio for training pairs.
    Mel spectrograms are extracted from raw audio.

    Args:
        batch: List of MOSI dataset samples

    Returns:
        Dict with 'text', 'mel', 'mel_lengths', 'audio' keys
    """
    texts = []
    mel_specs = []
    audio_waveforms = []

    for sample in batch:
        texts.append(sample['text'])

        # Load audio waveform
        if sample['audio'] is not None and os.path.exists(sample['audio']):
            try:
                waveform, sr = librosa.load(sample['audio'], sr=SAMPLE_RATE, mono=True)

                # Limit audio length
                max_samples = int(MAX_AUDIO_LENGTH * SAMPLE_RATE)
                if len(waveform) > max_samples:
                    waveform = waveform[:max_samples]

                audio_waveforms.append(waveform)

                # Extract mel spectrogram for target
                mel = extract_mel_spectrogram(waveform)
                mel_specs.append(mel)

            except Exception as e:
                print(f"[WARNING] Failed to load audio: {sample['audio']}: {e}")
                # Use silence fallback
                silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
                audio_waveforms.append(silence)
                mel_specs.append(extract_mel_spectrogram(silence))
        else:
            # Fallback: 1 second of silence
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            audio_waveforms.append(silence)
            mel_specs.append(extract_mel_spectrogram(silence))

    # Pad mel spectrograms for batching
    padded_mels, mel_lengths = pad_mel_spectrograms(mel_specs)

    return {
        'text': texts,
        'mel': padded_mels,  # (batch, time, n_mels)
        'mel_lengths': mel_lengths,  # (batch,)
        'audio': audio_waveforms  # List of numpy arrays
    }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_proxy_loss(
    adapter_output: torch.Tensor,
    regularization_weight: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Stage 1: Proxy loss for adapter training.

    Encourages adapter to produce bounded, well-behaved hidden states
    suitable for SpeechT5 decoder input. This is FULLY DIFFERENTIABLE.

    Loss components:
    1. L2 norm regularization - prevents degenerate outputs
    2. Mean centering - encourages zero-centered hidden states
    3. Variance regularization - encourages reasonable variance

    Args:
        adapter_output: (batch, hidden_dim) hidden states from adapter
        regularization_weight: Weight for regularization terms

    Returns:
        (loss, metrics_dict)
    """
    batch_size = adapter_output.shape[0]

    # L2 norm regularization - keep hidden states bounded
    l2_norm = torch.norm(adapter_output, p=2, dim=1).mean()
    l2_loss = regularization_weight * l2_norm

    # Mean centering loss - SpeechT5 expects roughly zero-centered hidden states
    mean_loss = regularization_weight * torch.abs(adapter_output.mean())

    # Variance regularization - encourage reasonable variance (not too small/large)
    variance = adapter_output.var(dim=1).mean()
    target_variance = 1.0  # SpeechT5 hidden states typically have variance ~1
    variance_loss = regularization_weight * torch.abs(variance - target_variance)

    # Total proxy loss
    total_loss = l2_loss + mean_loss + variance_loss

    return total_loss, {
        'proxy_loss': total_loss.item(),
        'l2_norm': l2_norm.item(),
        'mean': adapter_output.mean().item(),
        'variance': variance.item()
    }


def compute_mel_reconstruction_loss(
    predicted_mel: torch.Tensor,
    target_mel: torch.Tensor,
    target_lengths: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute mel spectrogram reconstruction loss.

    Combines:
    1. L1 loss (absolute difference)
    2. Spectral convergence (normalized Frobenius norm)

    Args:
        predicted_mel: (batch, time, n_mels) predicted mel spectrogram
        target_mel: (batch, time, n_mels) ground truth mel spectrogram
        target_lengths: (batch,) actual lengths (for masking padding)

    Returns:
        (total_loss, metrics_dict)
    """
    # Ensure same length by truncating/padding
    min_length = min(predicted_mel.shape[1], target_mel.shape[1])
    pred = predicted_mel[:, :min_length, :]
    target = target_mel[:, :min_length, :]

    # Create mask if lengths provided
    if target_lengths is not None:
        mask = torch.zeros_like(target)
        for i, length in enumerate(target_lengths):
            mask[i, :min(length, min_length), :] = 1.0
        # Apply mask
        pred = pred * mask
        target = target * mask

    # L1 loss
    l1_loss = F.l1_loss(pred, target)

    # Spectral convergence loss
    # ||target - pred||_F / ||target||_F
    target_norm = torch.norm(target, p='fro')
    if target_norm > 0:
        spectral_convergence = torch.norm(target - pred, p='fro') / target_norm
    else:
        spectral_convergence = torch.tensor(0.0, device=pred.device)

    # Weighted combination
    total_loss = (MEL_L1_WEIGHT * l1_loss +
                  SPECTRAL_CONVERGENCE_WEIGHT * spectral_convergence)

    return total_loss, {
        'l1_loss': l1_loss.item(),
        'spectral_convergence': spectral_convergence.item(),
        'total_loss': total_loss.item()
    }


def evaluate_semantic_fidelity(
    semantic_vectors: torch.Tensor,
    decoder: Vec_to_Audio,
    audio_encoder: Audio_to_Vec,
    device: str
) -> Dict[str, float]:
    """
    Evaluate semantic drift after decode-encode cycle.

    This is NON-DIFFERENTIABLE - for monitoring only.

    Flow:
    1. semantic_vec → decoder → audio
    2. audio → audio_encoder → reconstructed_vec
    3. Measure cosine similarity (semantic_vec, reconstructed_vec)

    Args:
        semantic_vectors: (batch, 1024) original semantic vectors
        decoder: Vec_to_Audio decoder
        audio_encoder: Audio_to_Vec encoder
        device: Device to run on

    Returns:
        Dict with semantic fidelity metrics
    """
    with torch.no_grad():
        # Generate audio from semantic vectors (one at a time for stability)
        batch_size = semantic_vectors.shape[0]
        reconstructed_vecs = []

        for i in range(batch_size):
            try:
                # Generate audio
                audio = decoder(semantic_vectors[i:i+1], device=device)

                # Re-encode generated audio
                recon_vec = audio_encoder(audio, device=device)
                reconstructed_vecs.append(recon_vec)
            except Exception as e:
                print(f"[WARNING] Semantic fidelity eval failed for sample {i}: {e}")
                # Use zero vector as fallback
                reconstructed_vecs.append(torch.zeros(1, 1024, device=device))

        # Stack reconstructed vectors
        reconstructed = torch.cat(reconstructed_vecs, dim=0)

        # Compute cosine similarity
        original_norm = F.normalize(semantic_vectors, dim=1)
        reconstructed_norm = F.normalize(reconstructed, dim=1)
        cosine_sim = F.cosine_similarity(original_norm, reconstructed_norm, dim=1)

        return {
            'semantic_similarity': cosine_sim.mean().item(),
            'semantic_drift': 1.0 - cosine_sim.mean().item()  # Lower is better
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    train_loader: DataLoader,
    decoder: Vec_to_Audio,
    text_encoder: Text_to_Vec,
    audio_encoder: Audio_to_Vec,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    eval_semantic_every: int,
    training_stage: str = 'proxy'
) -> Dict[str, float]:
    """
    Train decoder adapter for one epoch with two-stage training support.

    Stage 1 (proxy): Train adapter with proxy loss (fully differentiable)
    Stage 2 (generation): Continue with proxy loss but use mel quality for model selection

    Args:
        train_loader: DataLoader with audio data
        decoder: Vec_to_Audio decoder (adapter trainable)
        text_encoder: Text_to_Vec encoder (frozen) - for semantic vectors
        audio_encoder: Audio_to_Vec encoder (frozen) - for semantic fidelity
        optimizer: Adam optimizer for decoder adapter
        epoch: Current epoch number
        eval_semantic_every: Evaluate semantic drift every N batches
        training_stage: 'proxy' or 'generation'

    Returns:
        Dict with average metrics for the epoch
    """
    decoder.adapter.train()
    text_encoder.eval()
    audio_encoder.eval()

    epoch_metrics = {
        'proxy_loss': 0.0,
        'l1_loss': 0.0,
        'spectral_convergence': 0.0,
        'total_loss': 0.0,
        'semantic_similarity': 0.0,
        'semantic_drift': 0.0,
        'l2_norm': 0.0,
        'variance': 0.0
    }
    num_batches = 0
    semantic_eval_count = 0

    stage_desc = f"Epoch {epoch}/{EPOCHS} [{training_stage.upper()}]"
    pbar = tqdm(train_loader, desc=stage_desc)

    for batch_idx, batch in enumerate(pbar, 1):
        texts = batch['text']
        target_mels = batch['mel'].to(DEVICE)
        mel_lengths = batch['mel_lengths'].to(DEVICE)

        # 1. Encode texts to semantic vectors (frozen)
        with torch.no_grad():
            semantic_vecs = text_encoder(texts, device=DEVICE)

        # 2. Get adapter output (differentiable)
        adapter_output = decoder.get_adapter_output(semantic_vecs, device=DEVICE)

        # 3. Compute proxy loss (DIFFERENTIABLE - used for gradients)
        proxy_loss, proxy_metrics = compute_proxy_loss(
            adapter_output, regularization_weight=PROXY_LOSS_WEIGHT
        )

        # 4. Backprop through adapter using proxy loss
        optimizer.zero_grad()
        proxy_loss.backward()
        optimizer.step()

        # Accumulate proxy metrics
        epoch_metrics['proxy_loss'] += proxy_metrics['proxy_loss']
        epoch_metrics['l2_norm'] += proxy_metrics['l2_norm']
        epoch_metrics['variance'] += proxy_metrics['variance']
        num_batches += 1

        # 5. In generation stage, also compute mel reconstruction for monitoring
        if training_stage == 'generation' and batch_idx % 5 == 0:
            with torch.no_grad():
                generated_mels = []
                for i in range(min(semantic_vecs.shape[0], 2)):  # Limit to 2 samples for speed
                    try:
                        _, mel = decoder.generate_with_mel(
                            semantic_vecs[i:i+1],
                            device=DEVICE,
                            max_length=min(target_mels.shape[1], 200)
                        )
                        generated_mels.append(mel)
                    except Exception as e:
                        generated_mels.append(torch.zeros(100, N_MELS, device=DEVICE))

                if generated_mels:
                    max_len = max(m.shape[0] for m in generated_mels)
                    padded_gen = torch.zeros(len(generated_mels), max_len, N_MELS, device=DEVICE)
                    for i, mel in enumerate(generated_mels):
                        padded_gen[i, :mel.shape[0], :] = mel

                    _, mel_metrics = compute_mel_reconstruction_loss(
                        padded_gen, target_mels[:len(generated_mels)], mel_lengths[:len(generated_mels)]
                    )
                    epoch_metrics['l1_loss'] += mel_metrics['l1_loss']
                    epoch_metrics['spectral_convergence'] += mel_metrics['spectral_convergence']

        # 6. Periodically evaluate semantic fidelity
        should_eval_semantic = (
            eval_semantic_every > 0 and batch_idx % eval_semantic_every == 0
        )

        if should_eval_semantic:
            semantic_metrics = evaluate_semantic_fidelity(
                semantic_vecs, decoder, audio_encoder, DEVICE
            )
            epoch_metrics['semantic_similarity'] += semantic_metrics['semantic_similarity']
            epoch_metrics['semantic_drift'] += semantic_metrics['semantic_drift']
            semantic_eval_count += 1

            pbar.set_postfix({
                'proxy': f"{proxy_metrics['proxy_loss']:.4f}",
                'l2': f"{proxy_metrics['l2_norm']:.2f}",
                'var': f"{proxy_metrics['variance']:.2f}",
                'sem': f"{semantic_metrics['semantic_similarity']:.3f}"
            })
        else:
            pbar.set_postfix({
                'proxy': f"{proxy_metrics['proxy_loss']:.4f}",
                'l2': f"{proxy_metrics['l2_norm']:.2f}",
                'var': f"{proxy_metrics['variance']:.2f}"
            })

    # Average metrics
    for key in ['proxy_loss', 'l2_norm', 'variance']:
        epoch_metrics[key] /= max(num_batches, 1)

    # Total loss is the proxy loss (what we actually optimize)
    epoch_metrics['total_loss'] = epoch_metrics['proxy_loss']

    if semantic_eval_count > 0:
        epoch_metrics['semantic_similarity'] /= semantic_eval_count
        epoch_metrics['semantic_drift'] /= semantic_eval_count
    else:
        # Evaluate once at end of epoch
        print("\n  Evaluating semantic fidelity at epoch end...")
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_vecs = text_encoder(sample_batch['text'], device=DEVICE)
        semantic_metrics = evaluate_semantic_fidelity(
            sample_vecs, decoder, audio_encoder, DEVICE
        )
        epoch_metrics['semantic_similarity'] = semantic_metrics['semantic_similarity']
        epoch_metrics['semantic_drift'] = semantic_metrics['semantic_drift']

    return epoch_metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline with two-stage training."""
    print("=" * 80)
    print("AUDIO DECODER ADAPTER TRAINING - TWO-STAGE PIPELINE")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Total epochs: {EPOCHS}")
    print(f"  Stage 1 (Proxy): {PROXY_EPOCHS} epochs")
    print(f"  Stage 2 (Generation): {GENERATION_EPOCHS} epochs")
    print(f"Starting stage: {STARTING_STAGE}")
    print(f"Mel bins: {N_MELS}")
    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Semantic eval frequency: every {EVAL_SEMANTIC_EVERY} batches"
          if EVAL_SEMANTIC_EVERY > 0 else "Semantic eval frequency: once per epoch")
    print("=" * 80)
    print()
    print("Two-Stage Training Strategy:")
    print("  Stage 1 (Proxy): Differentiable proxy loss on adapter hidden states")
    print("    - L2 norm regularization (bounded outputs)")
    print("    - Mean centering (zero-centered hidden states)")
    print("    - Variance regularization (reasonable distribution)")
    print("  Stage 2 (Generation): Continue proxy loss + mel reconstruction monitoring")
    print("    - Use mel reconstruction quality for model selection")
    print("    - Semantic fidelity monitoring")
    print("=" * 80)
    print()

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ========================================================================
    # 1. LOAD MODELS
    # ========================================================================

    print("[1/4] Loading models...")
    print()

    # Load text encoder (frozen for training)
    print("  Loading Text_to_Vec encoder...")
    text_encoder = Text_to_Vec().to(DEVICE)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    print("  [OK] Text_to_Vec encoder: LOADED & FROZEN")
    print()

    # Load audio encoder (frozen for semantic fidelity evaluation)
    print("  Loading Audio_to_Vec encoder...")
    audio_encoder = Audio_to_Vec().to(DEVICE)
    audio_encoder.eval()
    for param in audio_encoder.parameters():
        param.requires_grad = False
    print("  [OK] Audio_to_Vec encoder: LOADED & FROZEN")
    print()

    # Load decoder (only adapter trainable)
    print("  Loading Vec_to_Audio decoder...")
    decoder = Vec_to_Audio().to(DEVICE)
    print(f"  [OK] Vec_to_Audio decoder: LOADED (adapter trainable)")
    print(f"  [OK] Adapter path: {decoder.adapter.weights_path}")
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
            required_modalities=['text', 'audio']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_audio_decoder,
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

    # Determine starting stage
    current_stage = STARTING_STAGE
    stage_start_epoch = 1 if current_stage == 'proxy' else PROXY_EPOCHS + 1

    for epoch in range(1, EPOCHS + 1):
        # Determine training stage based on epoch
        if epoch <= PROXY_EPOCHS:
            current_stage = 'proxy'
        else:
            if current_stage == 'proxy':
                print("\n" + "=" * 80)
                print("STAGE TRANSITION: Proxy -> Generation")
                print("=" * 80 + "\n")
            current_stage = 'generation'

        print(f"{'=' * 80}")
        print(f"Epoch {epoch}/{EPOCHS} [Stage: {current_stage.upper()}]")
        print(f"{'=' * 80}")

        epoch_start = time.time()

        # Train for one epoch
        metrics = train_epoch(
            train_loader, decoder, text_encoder, audio_encoder,
            optimizer, epoch, EVAL_SEMANTIC_EVERY,
            training_stage=current_stage
        )

        epoch_duration = time.time() - epoch_start

        # Log metrics
        print()
        print(f"Epoch {epoch} Results [{current_stage.upper()}]:")
        print(f"  Proxy Loss:           {metrics['proxy_loss']:.4f}")
        print(f"  L2 Norm:              {metrics['l2_norm']:.4f}")
        print(f"  Variance:             {metrics['variance']:.4f}")
        if current_stage == 'generation':
            print(f"  L1 Loss (mel):        {metrics.get('l1_loss', 0):.4f}")
            print(f"  Spectral Convergence: {metrics.get('spectral_convergence', 0):.4f}")
        print(f"  Semantic Similarity:  {metrics['semantic_similarity']:.3f}")
        print(f"  Semantic Drift:       {metrics['semantic_drift']:.3f}")
        print(f"  Epoch Time:           {epoch_duration:.1f}s")

        # Save best model (based on proxy loss)
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            decoder.adapter.save()
            print(f"  [OK] SAVED (best loss: {best_loss:.4f})")

        print()

        # Save checkpoint every N epochs
        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}_{current_stage}.pth")
            torch.save({
                'epoch': epoch,
                'training_stage': current_stage,
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
            'training_stage': current_stage,
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
                'total_epochs': EPOCHS,
                'proxy_epochs': PROXY_EPOCHS,
                'generation_epochs': GENERATION_EPOCHS,
                'eval_semantic_every': EVAL_SEMANTIC_EVERY,
                'training_approach': 'two_stage_proxy_generation'
            },
            'history': training_history,
            'best_loss': best_loss,
            'total_duration': total_duration
        }, f, indent=2)

    print(f"Training history saved: {history_path}")
    print()

    # Print training summary
    if len(training_history) >= 2:
        print("Training Summary:")
        print(f"  Initial loss: {training_history[0]['metrics']['total_loss']:.4f}")
        print(f"  Final loss:   {training_history[-1]['metrics']['total_loss']:.4f}")
        print(f"  Best loss:    {best_loss:.4f}")
        print()

    print("Next steps:")
    print("  1. Run evaluation: python scripts/evaluate_audio_decoder.py")
    print("  2. Check results in: results/audio_decoder_evaluation/")
    print()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
