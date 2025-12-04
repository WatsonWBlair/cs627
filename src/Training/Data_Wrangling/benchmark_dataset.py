"""
Unified Benchmark Dataset for Training

This module provides a unified dataset class that combines data from multiple sources:
- CMU-MOSI (multimodal: text, audio, video)
- MTEB (text pairs from retrieval, STS, clustering tasks)
- GLUE (text pairs from NLI, paraphrase, QA tasks)

The dataset supports both cross-modal and text-only contrastive learning.

Usage:
    from src.Training.Data_Wrangling.benchmark_dataset import BenchmarkDataset
    
    dataset = BenchmarkDataset(
        data_sources=['mosi', 'mteb', 'glue'],
        split='train',
        modalities=['text', 'audio']
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
import numpy as np

# Try to import modality-specific dependencies
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not available. Install with: pip install librosa soundfile")

try:
    from PIL import Image
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install pillow")


class BenchmarkDataset(Dataset):
    """
    Unified dataset combining multiple benchmark data sources.
    
    Supports:
    - Cross-modal triplets (text-audio-video from MOSI)
    - Text-only triplets (from MTEB, GLUE)
    - Mixed training with configurable sampling ratios
    """
    
    def __init__(
        self,
        data_sources: List[str] = ['mosi', 'mteb', 'glue'],
        split: str = 'train',
        modalities: List[str] = ['text'],
        mosi_data_path: str = 'data/cmumosi/mosi/',
        mosi_audio_dir: str = 'data/cmumosi/audio/',
        mosi_video_dir: str = 'data/cmumosi/frames/',
        mteb_data_path: str = 'data/mteb/',
        glue_data_path: str = 'data/glue/',
        max_text_length: int = 512,
        sampling_weights: Dict[str, float] = None,
        cache_embeddings: bool = False
    ):
        """
        Initialize unified benchmark dataset.
        
        Args:
            data_sources: Which data sources to include ('mosi', 'mteb', 'glue')
            split: Dataset split ('train', 'valid', 'test')
            modalities: Which modalities to load ('text', 'audio', 'video')
            mosi_data_path: Path to MOSI metadata
            mosi_audio_dir: Path to MOSI audio files
            mosi_video_dir: Path to MOSI video frames
            mteb_data_path: Path to MTEB wrangled data
            glue_data_path: Path to GLUE wrangled data
            max_text_length: Maximum text sequence length
            sampling_weights: Relative sampling weights for each data source
            cache_embeddings: Whether to cache processed embeddings
        """
        self.data_sources = data_sources
        self.split = split
        self.modalities = modalities
        self.max_text_length = max_text_length
        self.cache_embeddings = cache_embeddings
        
        # Set default sampling weights
        if sampling_weights is None:
            # Equal weight to each source
            sampling_weights = {source: 1.0 for source in data_sources}
        self.sampling_weights = sampling_weights
        
        # Storage for different data types
        self.triplets = []  # (anchor, positive, negative) tuples
        self.multimodal_samples = []  # Samples with audio/video
        self.text_only_samples = []  # Text-only samples
        
        # Load data from each source
        if 'mosi' in data_sources and 'text' in modalities:
            self._load_mosi_data(mosi_data_path, mosi_audio_dir, mosi_video_dir)
        
        if 'mteb' in data_sources:
            self._load_mteb_data(mteb_data_path)
        
        if 'glue' in data_sources:
            self._load_glue_data(glue_data_path)
        
        # Create unified sample list
        self._create_unified_samples()
        
        print(f"Initialized BenchmarkDataset:")
        print(f"  Total samples: {len(self)}")
        print(f"  Data sources: {data_sources}")
        print(f"  Modalities: {modalities}")
        print(f"  Split: {split}")
        
    def _load_mosi_data(self, data_path: str, audio_dir: str, video_dir: str):
        """Load CMU-MOSI multimodal data."""
        try:
            # Import MOSI dataset
            from .mosi_dataset import MOSIRawVideoDataset
            
            print("Loading MOSI data...")
            mosi_dataset = MOSIRawVideoDataset(
                split=self.split,
                mosi_data_path=data_path,
                audio_dir=audio_dir,
                video_dir=video_dir
            )
            
            # Convert MOSI samples to triplets
            # For MOSI, we'll create pseudo-triplets by grouping by sentiment
            samples_by_sentiment = {'positive': [], 'negative': [], 'neutral': []}
            
            for i in range(len(mosi_dataset)):
                sample = mosi_dataset[i]
                
                # Determine sentiment category
                if sample['label'] > 0.5:
                    category = 'positive'
                elif sample['label'] < -0.5:
                    category = 'negative'
                else:
                    category = 'neutral'
                
                samples_by_sentiment[category].append(sample)
            
            # Create triplets from sentiment groups
            for category in samples_by_sentiment:
                samples = samples_by_sentiment[category]
                
                for i, anchor_sample in enumerate(samples):
                    # Positive: same sentiment
                    if len(samples) > 1:
                        positive_sample = random.choice([s for j, s in enumerate(samples) if j != i])
                    else:
                        positive_sample = anchor_sample
                    
                    # Negative: different sentiment
                    other_categories = [c for c in samples_by_sentiment if c != category]
                    if other_categories:
                        neg_category = random.choice(other_categories)
                        if samples_by_sentiment[neg_category]:
                            negative_sample = random.choice(samples_by_sentiment[neg_category])
                        else:
                            negative_sample = anchor_sample
                    else:
                        negative_sample = anchor_sample
                    
                    # Store as multimodal sample
                    self.multimodal_samples.append({
                        'anchor': anchor_sample,
                        'positive': positive_sample,
                        'negative': negative_sample,
                        'source': 'mosi'
                    })
            
            print(f"  Loaded {len(self.multimodal_samples)} MOSI samples")
            
        except Exception as e:
            print(f"  Error loading MOSI data: {e}")
    
    def _load_mteb_data(self, data_path: str):
        """Load MTEB text triplets."""
        triplets_file = Path(data_path) / "mteb_triplets.pkl"
        
        if triplets_file.exists():
            print("Loading MTEB data...")
            with open(triplets_file, 'rb') as f:
                mteb_triplets = pickle.load(f)
            
            # Convert to our format
            for anchor, positive, negative in mteb_triplets:
                self.text_only_samples.append({
                    'anchor': {'text': anchor},
                    'positive': {'text': positive},
                    'negative': {'text': negative},
                    'source': 'mteb'
                })
            
            print(f"  Loaded {len(mteb_triplets)} MTEB triplets")
        else:
            print(f"  MTEB data not found at {triplets_file}")
            print("  Run: python scripts/data_wrangling/wrangle_mteb_data.py")
    
    def _load_glue_data(self, data_path: str):
        """Load GLUE text triplets."""
        triplets_file = Path(data_path) / "glue_triplets.pkl"
        
        if triplets_file.exists():
            print("Loading GLUE data...")
            with open(triplets_file, 'rb') as f:
                glue_triplets = pickle.load(f)
            
            # Convert to our format
            for anchor, positive, negative in glue_triplets:
                self.text_only_samples.append({
                    'anchor': {'text': anchor},
                    'positive': {'text': positive},
                    'negative': {'text': negative},
                    'source': 'glue'
                })
            
            print(f"  Loaded {len(glue_triplets)} GLUE triplets")
        else:
            print(f"  GLUE data not found at {triplets_file}")
            print("  Run: python scripts/data_wrangling/wrangle_glue_data.py")
    
    def _create_unified_samples(self):
        """Create unified sample list with weighted sampling."""
        self.all_samples = []
        
        # Add samples based on weights
        source_samples = {
            'mosi': self.multimodal_samples,
            'mteb': [s for s in self.text_only_samples if s['source'] == 'mteb'],
            'glue': [s for s in self.text_only_samples if s['source'] == 'glue']
        }
        
        # Normalize weights
        total_weight = sum(self.sampling_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in self.sampling_weights.items()}
        else:
            # If no weights, return empty or uniform
            normalized_weights = {}
        
        # Calculate samples per source
        min_samples = min(len(samples) for samples in source_samples.values() if samples)
        
        for source, samples in source_samples.items():
            if source in self.data_sources and samples:
                # Sample proportionally
                n_samples = int(min_samples * normalized_weights.get(source, 1.0))
                n_samples = min(n_samples, len(samples))
                
                if n_samples > 0:
                    sampled = random.sample(samples, n_samples) if n_samples < len(samples) else samples
                    self.all_samples.extend(sampled)
        
        # Shuffle combined samples
        random.shuffle(self.all_samples)
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            Dict with keys:
                - anchor_text, positive_text, negative_text: Text strings
                - anchor_audio, positive_audio, negative_audio: Audio tensors (if available)
                - anchor_video, positive_video, negative_video: Image tensors (if available)
                - source: Data source name
        """
        sample = self.all_samples[idx]
        output = {'source': sample['source']}
        
        # Extract text (always available)
        output['anchor_text'] = sample['anchor'].get('text', '')
        output['positive_text'] = sample['positive'].get('text', '')
        output['negative_text'] = sample['negative'].get('text', '')
        
        # Extract audio if available and requested
        if 'audio' in self.modalities and sample['source'] == 'mosi':
            output['anchor_audio'] = self._load_audio(sample['anchor'].get('audio'))
            output['positive_audio'] = self._load_audio(sample['positive'].get('audio'))
            output['negative_audio'] = self._load_audio(sample['negative'].get('audio'))
        
        # Extract video/image if available and requested
        if 'video' in self.modalities and sample['source'] == 'mosi':
            output['anchor_video'] = self._load_image(sample['anchor'].get('video'))
            output['positive_video'] = self._load_image(sample['positive'].get('video'))
            output['negative_video'] = self._load_image(sample['negative'].get('video'))
        
        return output
    
    def _load_audio(self, audio_path: Optional[str]) -> Optional[torch.Tensor]:
        """Load audio file and convert to tensor."""
        if not audio_path or not AUDIO_AVAILABLE:
            return None
        
        try:
            if os.path.exists(audio_path):
                # Load audio at 16kHz for Whisper compatibility
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                return torch.FloatTensor(audio)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
        
        return None
    
    def _load_image(self, image_path: Optional[str]) -> Optional[torch.Tensor]:
        """Load image file and convert to tensor."""
        if not image_path or not IMAGE_AVAILABLE:
            return None
        
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                # Resize to standard size (224x224 for ViT)
                image = image.resize((224, 224))
                # Convert to tensor
                image_array = np.array(image).astype(np.float32) / 255.0
                return torch.FloatTensor(image_array).permute(2, 0, 1)  # CHW format
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
        
        return None


def collate_fn_benchmark(batch: List[Dict]) -> Dict[str, Union[List, torch.Tensor]]:
    """
    Custom collate function for benchmark dataset.
    
    Handles variable modalities and creates batched tensors.
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Batched dict with lists/tensors for each modality
    """
    collated = {
        'source': [sample['source'] for sample in batch]
    }
    
    # Collate text (always present)
    collated['anchor_text'] = [sample['anchor_text'] for sample in batch]
    collated['positive_text'] = [sample['positive_text'] for sample in batch]
    collated['negative_text'] = [sample['negative_text'] for sample in batch]
    
    # Collate audio if present
    if 'anchor_audio' in batch[0] and batch[0]['anchor_audio'] is not None:
        # Pad audio to same length
        max_len = max(sample['anchor_audio'].shape[0] for sample in batch 
                     if sample['anchor_audio'] is not None)
        
        anchor_audio = []
        positive_audio = []
        negative_audio = []
        
        for sample in batch:
            for key, target_list in [
                ('anchor_audio', anchor_audio),
                ('positive_audio', positive_audio),
                ('negative_audio', negative_audio)
            ]:
                audio = sample.get(key)
                if audio is not None:
                    # Pad or truncate
                    if audio.shape[0] < max_len:
                        audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[0]))
                    else:
                        audio = audio[:max_len]
                    target_list.append(audio)
                else:
                    target_list.append(torch.zeros(max_len))
        
        collated['anchor_audio'] = torch.stack(anchor_audio)
        collated['positive_audio'] = torch.stack(positive_audio)
        collated['negative_audio'] = torch.stack(negative_audio)
    
    # Collate video/image if present
    if 'anchor_video' in batch[0] and batch[0]['anchor_video'] is not None:
        anchor_video = []
        positive_video = []
        negative_video = []
        
        for sample in batch:
            for key, target_list in [
                ('anchor_video', anchor_video),
                ('positive_video', positive_video),
                ('negative_video', negative_video)
            ]:
                video = sample.get(key)
                if video is not None:
                    target_list.append(video)
                else:
                    # Use black image as placeholder
                    target_list.append(torch.zeros(3, 224, 224))
        
        collated['anchor_video'] = torch.stack(anchor_video)
        collated['positive_video'] = torch.stack(positive_video)
        collated['negative_video'] = torch.stack(negative_video)
    
    return collated


if __name__ == "__main__":
    """Test the dataset with sample data."""
    
    print("Testing BenchmarkDataset...")
    
    # Create dataset (text-only for testing)
    dataset = BenchmarkDataset(
        data_sources=['glue'],  # Start with GLUE only for testing
        split='train',
        modalities=['text'],
        glue_data_path='data/glue/'
    )
    
    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        print(f"\nSample from {sample['source']}:")
        print(f"  Anchor: {sample['anchor_text'][:100]}...")
        print(f"  Positive: {sample['positive_text'][:100]}...")
        print(f"  Negative: {sample['negative_text'][:100]}...")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn_benchmark
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch test:")
        print(f"  Batch size: {len(batch['anchor_text'])}")
        print(f"  Sources: {batch['source']}")
        
        print("\nDataset test completed successfully!")
    else:
        print("No data loaded. Please run data wrangling scripts first:")
        print("  python scripts/data_wrangling/wrangle_glue_data.py")
        print("  python scripts/data_wrangling/wrangle_mteb_data.py")