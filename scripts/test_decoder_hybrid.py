"""
Quick test script for hybrid decoder training validation.

Tests the teacher forcing + semantic fidelity approach with minimal hyperparameters.

Usage:
    python scripts/test_decoder_hybrid.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutput

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.Encoders import Text_to_Vec
from src.Decoders import Vec_to_Text
from src.Training.Data_Wrangling.mosi_dataset import MOSIRawVideoDataset

# Test hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
TEST_EPOCHS = 2

# Dataset paths
MOSI_DATA_PATH = "data/cmumosi/mosi/"
AUDIO_DIR = "data/cmumosi/audio/"
VIDEO_DIR = "data/cmumosi/frames/"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn_text_only(batch):
    """Extract only text from MOSI dataset batch."""
    return [sample['text'] for sample in batch]


def compute_teacher_forcing_loss(semantic_vectors, target_texts, adapter, bart_model, tokenizer, device):
    """Compute teacher forcing loss."""
    # Map semantic vectors to BART space
    adapter_output = adapter(semantic_vectors)

    # Create encoder outputs
    encoder_outputs = BaseModelOutput(
        last_hidden_state=adapter_output.unsqueeze(1),
        hidden_states=None,
        attentions=None
    )

    # Tokenize targets
    target_encoding = tokenizer(
        target_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Forward through BART with labels
    outputs = bart_model(
        encoder_outputs=encoder_outputs,
        labels=target_encoding.input_ids
    )

    return outputs.loss


def evaluate_semantic_fidelity(semantic_vectors, decoder, encoder, device):
    """Evaluate semantic drift (non-differentiable)."""
    import torch.nn.functional as F

    with torch.no_grad():
        # Generate text
        reconstructed_texts = decoder(semantic_vectors, device=device)
        if isinstance(reconstructed_texts, str):
            reconstructed_texts = [reconstructed_texts]

        # Re-encode
        reconstructed_vecs = encoder(reconstructed_texts, device=device)

        # Compute similarity
        original_norm = F.normalize(semantic_vectors, dim=1)
        reconstructed_norm = F.normalize(reconstructed_vecs, dim=1)
        cosine_sim = F.cosine_similarity(original_norm, reconstructed_norm, dim=1)

        return {
            'semantic_similarity': cosine_sim.mean().item(),
            'semantic_drift': 1.0 - cosine_sim.mean().item()
        }


def main():
    """Quick validation test."""
    print("=" * 80)
    print("DECODER HYBRID TRAINING - VALIDATION TEST")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Test epochs: {TEST_EPOCHS}")
    print("=" * 80)
    print()

    # Load models
    print("[1/4] Loading models...")
    encoder = Text_to_Vec().to(DEVICE)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    decoder = Vec_to_Text().to(DEVICE)
    for param in decoder.decoder.parameters():
        param.requires_grad = False

    # Load BART for teacher forcing
    from transformers import BartForConditionalGeneration, BartTokenizer
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(DEVICE)
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart_model.eval()
    for param in bart_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(decoder.adapter.parameters(), lr=LEARNING_RATE)

    trainable_params = sum(p.numel() for p in decoder.adapter.parameters() if p.requires_grad)
    print(f"  [OK] Models loaded")
    print(f"  [OK] Trainable parameters: {trainable_params:,}")
    print()

    # Load dataset
    print("[2/4] Loading dataset...")
    try:
        train_dataset = MOSIRawVideoDataset(
            split='train',
            mosi_data_path=MOSI_DATA_PATH,
            audio_dir=AUDIO_DIR,
            video_dir=VIDEO_DIR,
            required_modalities=['text']  # Text-only training
        )

        from torch.utils.data import Subset
        test_subset = Subset(train_dataset, range(min(20, len(train_dataset))))

        train_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_text_only,
            num_workers=0
        )

        print(f"  [OK] Loaded {len(test_subset)} samples for testing")
        print(f"  [OK] Batches: {len(train_loader)}")
        print()

    except FileNotFoundError as e:
        print(f"  [FAIL] Dataset not found: {e}")
        print()
        return 1

    # Test training loop
    print("[3/4] Running hybrid training test...")
    print()

    decoder.adapter.train()

    for epoch in range(1, TEST_EPOCHS + 1):
        print(f"Epoch {epoch}/{TEST_EPOCHS}")

        for batch_idx, batch_texts in enumerate(train_loader, 1):
            # Encode texts
            with torch.no_grad():
                semantic_vecs = encoder(batch_texts, device=DEVICE)

            # Teacher forcing loss (differentiable)
            tf_loss = compute_teacher_forcing_loss(
                semantic_vecs, batch_texts, decoder.adapter,
                bart_model, bart_tokenizer, DEVICE
            )

            # Backprop
            optimizer.zero_grad()
            tf_loss.backward()
            optimizer.step()

            # Evaluate semantic fidelity (non-differentiable, monitoring only)
            semantic_metrics = evaluate_semantic_fidelity(
                semantic_vecs, decoder, encoder, DEVICE
            )

            print(f"  Batch {batch_idx}/{len(train_loader)} - "
                  f"TF Loss: {tf_loss.item():.4f}, "
                  f"Sem Sim: {semantic_metrics['semantic_similarity']:.3f}, "
                  f"Drift: {semantic_metrics['semantic_drift']:.3f}")

        print()

    # Test generation
    print("[4/4] Testing generation quality...")
    decoder.eval()

    sample_texts = next(iter(train_loader))[:3]  # Get 3 samples
    with torch.no_grad():
        sample_vecs = encoder(sample_texts, device=DEVICE)
        generated_texts = decoder(sample_vecs, device=DEVICE)

    print("  Sample generations:")
    for i, (orig, gen) in enumerate(zip(sample_texts, generated_texts), 1):
        print(f"    [{i}] Original: {orig[:60]}...")
        print(f"        Generated: {gen[:60]}...")
        print()

    print("=" * 80)
    print("VALIDATION TEST PASSED")
    print("=" * 80)
    print()
    print("Hybrid approach validated:")
    print("  [OK] Teacher forcing loss is differentiable")
    print("  [OK] Semantic drift can be monitored")
    print("  [OK] Text generation works after training")
    print()
    print("Next steps:")
    print("  1. Run full training: python src/Training/train_text_decoder.py")
    print("  2. Monitor both teacher forcing loss and semantic drift")
    print()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
