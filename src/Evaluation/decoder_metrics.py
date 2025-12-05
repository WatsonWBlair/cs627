"""
Decoder Evaluation Metrics

This module provides evaluation metrics specifically for decoder modules
that translate semantic vectors back to their original modalities (text, audio, image).

Key Metrics:
1. BLEU Score - Precision-based text reconstruction quality (n-gram overlap)
2. ROUGE Score - Recall-based text reconstruction quality (n-gram overlap)
3. Semantic Fidelity - Cosine similarity between original and reconstructed vectors
4. Combined Reconstruction Quality - Weighted average of BLEU and ROUGE

Audio Metrics:
1. Mel Cepstral Distortion (MCD) - Spectral similarity between audio waveforms
2. Audio Semantic Fidelity - Cosine similarity after audio decode-encode cycle

Usage:
    from src.Evaluation.decoder_metrics import evaluate_text_decoder, evaluate_audio_decoder

    # Text decoder evaluation
    metrics = evaluate_text_decoder(
        decoder=Vec_to_Text(),
        encoder=Text_to_Vec(),
        test_texts=['example text', ...],
        device='cuda'
    )

    # Audio decoder evaluation
    audio_metrics = evaluate_audio_decoder(
        decoder=Vec_to_Audio(),
        text_encoder=Text_to_Vec(),
        audio_encoder=Audio_to_Vec(),
        test_texts=['example text', ...],
        device='cuda'
    )
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import evaluate


def compute_bleu_scores(
    predictions: List[str],
    references: List[str],
    max_order: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU scores for text reconstruction.

    BLEU (Bilingual Evaluation Understudy) measures precision of n-gram
    overlap between generated and reference texts.

    Args:
        predictions: Generated texts from decoder
        references: Ground truth reference texts
        max_order: Maximum n-gram order (default: 4)

    Returns:
        Dict with BLEU scores (overall score + individual n-gram precisions)

    Example:
        >>> predictions = ["the cat sat on the mat", "hello world"]
        >>> references = ["the cat is on the mat", "hello world"]
        >>> scores = compute_bleu_scores(predictions, references)
        >>> print(f"BLEU: {scores['bleu']:.3f}")
    """
    bleu_metric = evaluate.load("bleu")

    # Format references as list of lists (required by evaluate library)
    formatted_refs = [[ref] for ref in references]

    results = bleu_metric.compute(
        predictions=predictions,
        references=formatted_refs,
        max_order=max_order
    )

    return {
        'bleu': results.get('bleu', 0.0),
        'bleu_1': results.get('precisions', [0.0])[0] if results.get('precisions') else 0.0,
        'bleu_2': results.get('precisions', [0.0, 0.0])[1] if len(results.get('precisions', [])) > 1 else 0.0,
        'bleu_3': results.get('precisions', [0.0, 0.0, 0.0])[2] if len(results.get('precisions', [])) > 2 else 0.0,
        'bleu_4': results.get('precisions', [0.0, 0.0, 0.0, 0.0])[3] if len(results.get('precisions', [])) > 3 else 0.0,
    }


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']
) -> Dict[str, float]:
    """
    Compute ROUGE scores for text reconstruction.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    recall-based n-gram and sequence overlap.

    Args:
        predictions: Generated texts from decoder
        references: Ground truth reference texts
        rouge_types: Which ROUGE variants to compute

    Returns:
        Dict with ROUGE scores (precision, recall, F1 for each type)

    Example:
        >>> predictions = ["the cat sat on mat"]
        >>> references = ["the cat is on the mat"]
        >>> scores = compute_rouge_scores(predictions, references)
        >>> print(f"ROUGE-L: {scores['rougeL']:.3f}")
    """
    rouge_metric = evaluate.load("rouge")

    results = rouge_metric.compute(
        predictions=predictions,
        references=references,
        rouge_types=rouge_types,
        use_stemmer=True
    )

    # Extract F1 scores (primary metric)
    return {
        'rouge1': results.get('rouge1', 0.0),
        'rouge2': results.get('rouge2', 0.0),
        'rougeL': results.get('rougeL', 0.0),
    }


def compute_semantic_fidelity(
    original_vectors: torch.Tensor,
    reconstructed_vectors: torch.Tensor
) -> Dict[str, float]:
    """
    Measure semantic drift via cosine similarity.

    Computes how well the reconstructed text (after re-encoding) matches
    the original semantic vector. High similarity = low semantic drift.

    Args:
        original_vectors: (N, D) original semantic vectors before decoding
        reconstructed_vectors: (N, D) semantic vectors after decode -> re-encode

    Returns:
        Dict with similarity metrics (mean, std, min, max)

    Example:
        >>> original = torch.randn(100, 1024)
        >>> reconstructed = original + 0.1 * torch.randn(100, 1024)
        >>> fidelity = compute_semantic_fidelity(original, reconstructed)
        >>> print(f"Mean similarity: {fidelity['mean_similarity']:.3f}")
    """
    # Normalize vectors
    original_norm = F.normalize(original_vectors, p=2, dim=1)
    reconstructed_norm = F.normalize(reconstructed_vectors, p=2, dim=1)

    # Compute cosine similarity for each pair
    similarities = torch.sum(original_norm * reconstructed_norm, dim=1)

    return {
        'mean_similarity': similarities.mean().item(),
        'std_similarity': similarities.std().item(),
        'min_similarity': similarities.min().item(),
        'max_similarity': similarities.max().item(),
        'median_similarity': similarities.median().item()
    }


def compute_reconstruction_quality(
    bleu_scores: Dict[str, float],
    rouge_scores: Dict[str, float],
    bleu_weight: float = 0.5,
    rouge_weight: float = 0.5
) -> float:
    """
    Combine BLEU and ROUGE into overall reconstruction quality score.

    Args:
        bleu_scores: Dict from compute_bleu_scores()
        rouge_scores: Dict from compute_rouge_scores()
        bleu_weight: Weight for BLEU score (default: 0.5)
        rouge_weight: Weight for ROUGE score (default: 0.5)

    Returns:
        Combined reconstruction quality in [0, 1] (higher is better)
    """
    bleu = bleu_scores.get('bleu', 0.0)
    rouge_l = rouge_scores.get('rougeL', 0.0)

    return bleu_weight * bleu + rouge_weight * rouge_l


def evaluate_text_decoder(
    decoder,
    encoder,
    test_texts: List[str],
    device: str = 'cuda',
    batch_size: int = 32,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Comprehensive evaluation of text decoder (Vec_to_Text).

    Evaluation pipeline:
    1. Encode test texts to semantic vectors (encoder frozen)
    2. Decode vectors back to text (decoder)
    3. Re-encode reconstructed text to vectors (encoder frozen)
    4. Compute BLEU, ROUGE, and semantic fidelity metrics

    Args:
        decoder: Vec_to_Text decoder module
        encoder: Text_to_Vec encoder module (frozen)
        test_texts: List of test text samples
        device: 'cuda' or 'cpu'
        batch_size: Batch size for processing
        verbose: Print progress messages

    Returns:
        Dict with all computed metrics

    Example:
        >>> from src.Encoders import Text_to_Vec
        >>> from src.Decoders import Vec_to_Text
        >>>
        >>> decoder = Vec_to_Text()
        >>> encoder = Text_to_Vec()
        >>> test_texts = ["example text 1", "example text 2", ...]
        >>>
        >>> metrics = evaluate_text_decoder(decoder, encoder, test_texts)
        >>> print(f"BLEU: {metrics['bleu']['bleu']:.3f}")
        >>> print(f"ROUGE-L: {metrics['rouge']['rougeL']:.3f}")
        >>> print(f"Semantic fidelity: {metrics['semantic_fidelity']['mean_similarity']:.3f}")
    """
    if verbose:
        print("=" * 80)
        print("TEXT DECODER EVALUATION")
        print("=" * 80)
        print(f"Test samples: {len(test_texts)}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        print()

    decoder.eval()
    encoder.eval()

    all_original_vecs = []
    all_reconstructed_vecs = []
    all_reconstructed_texts = []

    # Process in batches
    num_batches = (len(test_texts) + batch_size - 1) // batch_size

    if verbose:
        print("[1/4] Encoding and decoding test samples...")
        from tqdm import tqdm
        batch_iter = tqdm(range(num_batches), desc="Processing batches")
    else:
        batch_iter = range(num_batches)

    with torch.no_grad():
        for batch_idx in batch_iter:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_texts))
            batch_texts = test_texts[start_idx:end_idx]

            # 1. Encode original texts
            original_vecs = encoder(batch_texts, device=device)

            # 2. Decode to text
            reconstructed_texts = decoder(original_vecs, device=device)

            # Handle single vs batch output
            if isinstance(reconstructed_texts, str):
                reconstructed_texts = [reconstructed_texts]

            # 3. Re-encode reconstructed texts
            reconstructed_vecs = encoder(reconstructed_texts, device=device)

            # Store results
            all_original_vecs.append(original_vecs)
            all_reconstructed_vecs.append(reconstructed_vecs)
            all_reconstructed_texts.extend(reconstructed_texts)

    # Concatenate all batches
    all_original_vecs = torch.cat(all_original_vecs, dim=0)
    all_reconstructed_vecs = torch.cat(all_reconstructed_vecs, dim=0)

    if verbose:
        print()
        print("[2/4] Computing BLEU scores...")

    # Compute BLEU
    bleu_scores = compute_bleu_scores(all_reconstructed_texts, test_texts)

    if verbose:
        print("[3/4] Computing ROUGE scores...")

    # Compute ROUGE
    rouge_scores = compute_rouge_scores(all_reconstructed_texts, test_texts)

    if verbose:
        print("[4/4] Computing semantic fidelity...")

    # Compute semantic fidelity
    semantic_fidelity = compute_semantic_fidelity(all_original_vecs, all_reconstructed_vecs)

    # Combined reconstruction quality
    reconstruction_quality = compute_reconstruction_quality(bleu_scores, rouge_scores)

    if verbose:
        print()
        print("Evaluation complete!")
        print()

    return {
        'bleu': bleu_scores,
        'rouge': rouge_scores,
        'semantic_fidelity': semantic_fidelity,
        'reconstruction_quality': reconstruction_quality,
        'num_samples': len(test_texts),
        'reconstructed_texts': all_reconstructed_texts  # For qualitative inspection
    }


def print_decoder_metrics_summary(metrics: Dict[str, any], show_examples: int = 3):
    """
    Pretty-print decoder evaluation metrics.

    Args:
        metrics: Output from evaluate_text_decoder()
        show_examples: Number of example reconstructions to display
    """
    print("=" * 80)
    print("DECODER EVALUATION SUMMARY")
    print("=" * 80)
    print()

    # Overall reconstruction quality
    print("OVERALL RECONSTRUCTION QUALITY")
    print("-" * 80)
    print(f"  Combined Score:  {metrics['reconstruction_quality']:.3f}")
    print()

    # BLEU scores
    print("BLEU SCORES (Precision-based)")
    print("-" * 80)
    bleu = metrics['bleu']
    print(f"  BLEU (overall):  {bleu['bleu']:.3f}")
    print(f"  BLEU-1 (1-gram): {bleu['bleu_1']:.3f}")
    print(f"  BLEU-2 (2-gram): {bleu['bleu_2']:.3f}")
    print(f"  BLEU-3 (3-gram): {bleu['bleu_3']:.3f}")
    print(f"  BLEU-4 (4-gram): {bleu['bleu_4']:.3f}")
    print()

    # ROUGE scores
    print("ROUGE SCORES (Recall-based)")
    print("-" * 80)
    rouge = metrics['rouge']
    print(f"  ROUGE-1:         {rouge['rouge1']:.3f}")
    print(f"  ROUGE-2:         {rouge['rouge2']:.3f}")
    print(f"  ROUGE-L:         {rouge['rougeL']:.3f}")
    print()

    # Semantic fidelity
    print("SEMANTIC FIDELITY (Cosine Similarity)")
    print("-" * 80)
    fidelity = metrics['semantic_fidelity']
    print(f"  Mean:            {fidelity['mean_similarity']:.3f}")
    print(f"  Median:          {fidelity['median_similarity']:.3f}")
    print(f"  Std Dev:         {fidelity['std_similarity']:.3f}")
    print(f"  Min:             {fidelity['min_similarity']:.3f}")
    print(f"  Max:             {fidelity['max_similarity']:.3f}")
    print()

    # Example reconstructions
    if show_examples > 0 and 'reconstructed_texts' in metrics:
        print("EXAMPLE RECONSTRUCTIONS")
        print("-" * 80)
        reconstructed = metrics['reconstructed_texts'][:show_examples]
        for i, text in enumerate(reconstructed, 1):
            # Truncate long texts
            display_text = text[:100] + "..." if len(text) > 100 else text
            print(f"  [{i}] {display_text}")
        print()

    print("=" * 80)


def save_decoder_metrics(
    metrics: Dict[str, any],
    output_path: str,
    include_reconstructions: bool = False
):
    """
    Save decoder evaluation metrics to JSON file.

    Args:
        metrics: Output from evaluate_text_decoder()
        output_path: Path to save JSON file
        include_reconstructions: Whether to include all reconstructed texts
    """
    import json

    # Prepare serializable metrics
    serializable_metrics = {
        'bleu': metrics['bleu'],
        'rouge': metrics['rouge'],
        'semantic_fidelity': metrics['semantic_fidelity'],
        'reconstruction_quality': metrics['reconstruction_quality'],
        'num_samples': metrics['num_samples']
    }

    # Optionally include reconstructed texts
    if include_reconstructions and 'reconstructed_texts' in metrics:
        serializable_metrics['reconstructed_texts'] = metrics['reconstructed_texts']

    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    print(f"[OK] Metrics saved to: {output_path}")


# ============================================================================
# AUDIO DECODER METRICS
# ============================================================================

def compute_mel_cepstral_distortion(
    generated_audio: np.ndarray,
    reference_audio: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13
) -> float:
    """
    Compute Mel Cepstral Distortion (MCD) between audio waveforms.

    MCD measures the spectral distance between two audio signals using
    Mel-frequency cepstral coefficients. Lower is better.

    Args:
        generated_audio: Generated audio waveform (samples,)
        reference_audio: Reference audio waveform (samples,)
        sample_rate: Audio sample rate (default: 16000)
        n_mfcc: Number of MFCCs to compute (default: 13)

    Returns:
        MCD value in dB (lower is better, 0 = identical)

    Note:
        - Typical values: 3-5 dB for good synthesis, >8 dB for poor synthesis
        - Requires librosa for MFCC computation
    """
    try:
        import librosa
    except ImportError:
        print("[WARNING] librosa not installed, MCD computation skipped")
        return float('inf')

    # Ensure same length (truncate to shorter)
    min_len = min(len(generated_audio), len(reference_audio))
    if min_len == 0:
        return float('inf')

    gen = generated_audio[:min_len]
    ref = reference_audio[:min_len]

    # Extract MFCCs
    gen_mfcc = librosa.feature.mfcc(y=gen.astype(np.float32), sr=sample_rate, n_mfcc=n_mfcc)
    ref_mfcc = librosa.feature.mfcc(y=ref.astype(np.float32), sr=sample_rate, n_mfcc=n_mfcc)

    # Align lengths (DTW would be better but this is simpler)
    min_frames = min(gen_mfcc.shape[1], ref_mfcc.shape[1])
    if min_frames == 0:
        return float('inf')

    gen_mfcc = gen_mfcc[:, :min_frames]
    ref_mfcc = ref_mfcc[:, :min_frames]

    # Compute MCD (excluding 0th coefficient which is energy)
    # MCD = (10 / ln(10)) * sqrt(2 * sum((gen_mfcc - ref_mfcc)^2))
    # Simplified: MCD = sqrt(sum((gen_mfcc - ref_mfcc)^2))
    diff = gen_mfcc[1:, :] - ref_mfcc[1:, :]  # Exclude 0th coefficient
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))

    # Convert to dB scale
    mcd_db = (10.0 / np.log(10)) * mcd

    return float(mcd_db)


def compute_audio_semantic_fidelity(
    original_vectors: torch.Tensor,
    decoder,
    audio_encoder,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure semantic drift for audio decoder via encode-decode-encode cycle.

    Flow:
    1. semantic_vec -> audio decoder -> audio waveform
    2. audio waveform -> audio encoder -> reconstructed semantic vector
    3. Compare original and reconstructed vectors via cosine similarity

    Args:
        original_vectors: (N, 1024) original semantic vectors
        decoder: Vec_to_Audio decoder
        audio_encoder: Audio_to_Vec encoder
        device: 'cuda' or 'cpu'

    Returns:
        Dict with semantic fidelity metrics
    """
    decoder.eval()
    audio_encoder.eval()

    similarities = []

    with torch.no_grad():
        for i in range(original_vectors.shape[0]):
            try:
                # Generate audio
                audio = decoder(original_vectors[i:i+1].to(device), device=device)

                # Re-encode audio
                reconstructed_vec = audio_encoder(audio, device=device)

                # Compute cosine similarity
                orig_norm = F.normalize(original_vectors[i:i+1].to(device), dim=1)
                recon_norm = F.normalize(reconstructed_vec, dim=1)
                sim = F.cosine_similarity(orig_norm, recon_norm, dim=1)
                similarities.append(sim.item())

            except Exception as e:
                print(f"[WARNING] Audio semantic fidelity failed for sample {i}: {e}")
                similarities.append(0.0)

    similarities = np.array(similarities)

    return {
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'median_similarity': float(np.median(similarities))
    }


def evaluate_audio_decoder(
    decoder,
    text_encoder,
    audio_encoder,
    test_texts: List[str],
    reference_audios: Optional[List[np.ndarray]] = None,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, any]:
    """
    Comprehensive evaluation of audio decoder (Vec_to_Audio).

    Evaluation pipeline:
    1. Encode test texts to semantic vectors (text encoder frozen)
    2. Decode vectors to audio (audio decoder)
    3. Re-encode generated audio to vectors (audio encoder frozen)
    4. Compute semantic fidelity and optionally MCD

    Args:
        decoder: Vec_to_Audio decoder module
        text_encoder: Text_to_Vec encoder module (frozen)
        audio_encoder: Audio_to_Vec encoder module (frozen)
        test_texts: List of test text samples to generate audio for
        reference_audios: Optional list of reference audio waveforms for MCD
        device: 'cuda' or 'cpu'
        verbose: Print progress messages

    Returns:
        Dict with all computed metrics

    Example:
        >>> from src.Encoders import Text_to_Vec, Audio_to_Vec
        >>> from src.Decoders import Vec_to_Audio
        >>>
        >>> decoder = Vec_to_Audio()
        >>> text_encoder = Text_to_Vec()
        >>> audio_encoder = Audio_to_Vec()
        >>> test_texts = ["hello world", "how are you", ...]
        >>>
        >>> metrics = evaluate_audio_decoder(decoder, text_encoder, audio_encoder, test_texts)
        >>> print(f"Semantic fidelity: {metrics['semantic_fidelity']['mean_similarity']:.3f}")
    """
    if verbose:
        print("=" * 80)
        print("AUDIO DECODER EVALUATION")
        print("=" * 80)
        print(f"Test samples: {len(test_texts)}")
        print(f"Device: {device}")
        print()

    decoder.eval()
    text_encoder.eval()
    audio_encoder.eval()

    generated_audios = []
    all_semantic_vecs = []

    if verbose:
        print("[1/3] Encoding texts and generating audio...")
        from tqdm import tqdm
        iterator = tqdm(enumerate(test_texts), total=len(test_texts), desc="Generating audio")
    else:
        iterator = enumerate(test_texts)

    with torch.no_grad():
        for i, text in iterator:
            try:
                # Encode text to semantic vector
                semantic_vec = text_encoder([text], device=device)
                all_semantic_vecs.append(semantic_vec)

                # Generate audio
                audio = decoder(semantic_vec, device=device)
                generated_audios.append(audio)

            except Exception as e:
                print(f"[WARNING] Audio generation failed for sample {i}: {e}")
                all_semantic_vecs.append(torch.zeros(1, 1024, device=device))
                generated_audios.append(np.zeros(16000, dtype=np.float32))

    # Stack semantic vectors
    all_semantic_vecs = torch.cat(all_semantic_vecs, dim=0)

    if verbose:
        print()
        print("[2/3] Computing semantic fidelity...")

    # Compute semantic fidelity
    semantic_fidelity = compute_audio_semantic_fidelity(
        all_semantic_vecs, decoder, audio_encoder, device
    )

    # Compute MCD if reference audios provided
    mcd_scores = None
    if reference_audios is not None and len(reference_audios) == len(generated_audios):
        if verbose:
            print("[3/3] Computing Mel Cepstral Distortion...")

        mcd_values = []
        for gen, ref in zip(generated_audios, reference_audios):
            mcd = compute_mel_cepstral_distortion(gen, ref)
            mcd_values.append(mcd)

        mcd_scores = {
            'mean_mcd': float(np.mean(mcd_values)),
            'std_mcd': float(np.std(mcd_values)),
            'min_mcd': float(np.min(mcd_values)),
            'max_mcd': float(np.max(mcd_values))
        }
    else:
        if verbose:
            print("[3/3] Skipping MCD (no reference audio provided)")

    if verbose:
        print()
        print("Evaluation complete!")
        print()

    return {
        'semantic_fidelity': semantic_fidelity,
        'mcd': mcd_scores,
        'num_samples': len(test_texts),
        'generated_audios': generated_audios  # For qualitative inspection
    }


def print_audio_decoder_metrics_summary(metrics: Dict[str, any]):
    """
    Pretty-print audio decoder evaluation metrics.

    Args:
        metrics: Output from evaluate_audio_decoder()
    """
    print("=" * 80)
    print("AUDIO DECODER EVALUATION SUMMARY")
    print("=" * 80)
    print()

    # Semantic fidelity
    print("SEMANTIC FIDELITY (Cosine Similarity)")
    print("-" * 80)
    fidelity = metrics['semantic_fidelity']
    print(f"  Mean:            {fidelity['mean_similarity']:.3f}")
    print(f"  Median:          {fidelity['median_similarity']:.3f}")
    print(f"  Std Dev:         {fidelity['std_similarity']:.3f}")
    print(f"  Min:             {fidelity['min_similarity']:.3f}")
    print(f"  Max:             {fidelity['max_similarity']:.3f}")
    print()

    # MCD if available
    if metrics.get('mcd') is not None:
        print("MEL CEPSTRAL DISTORTION (dB, lower is better)")
        print("-" * 80)
        mcd = metrics['mcd']
        print(f"  Mean MCD:        {mcd['mean_mcd']:.2f} dB")
        print(f"  Std MCD:         {mcd['std_mcd']:.2f} dB")
        print(f"  Min MCD:         {mcd['min_mcd']:.2f} dB")
        print(f"  Max MCD:         {mcd['max_mcd']:.2f} dB")
        print()
    else:
        print("MEL CEPSTRAL DISTORTION: Not computed (no reference audio)")
        print()

    print(f"Total samples evaluated: {metrics['num_samples']}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    # Example usage with mock data
    print("Testing decoder metrics with mock data...")
    print()

    # Mock predictions and references
    predictions = [
        "the cat sat on the mat",
        "hello world this is a test",
        "artificial intelligence is fascinating"
    ]

    references = [
        "the cat is sitting on the mat",
        "hello world this is a test",
        "artificial intelligence is very fascinating"
    ]

    # Test BLEU
    print("Computing BLEU scores...")
    bleu = compute_bleu_scores(predictions, references)
    print(f"  BLEU: {bleu['bleu']:.3f}")
    print()

    # Test ROUGE
    print("Computing ROUGE scores...")
    rouge = compute_rouge_scores(predictions, references)
    print(f"  ROUGE-L: {rouge['rougeL']:.3f}")
    print()

    # Test semantic fidelity with random vectors
    print("Computing semantic fidelity...")
    original_vecs = torch.randn(3, 1024)
    reconstructed_vecs = original_vecs + 0.1 * torch.randn(3, 1024)
    fidelity = compute_semantic_fidelity(original_vecs, reconstructed_vecs)
    print(f"  Mean similarity: {fidelity['mean_similarity']:.3f}")
    print()

    # Test reconstruction quality
    print("Computing reconstruction quality...")
    quality = compute_reconstruction_quality(bleu, rouge)
    print(f"  Combined quality: {quality:.3f}")
    print()

    print("Example completed successfully!")
