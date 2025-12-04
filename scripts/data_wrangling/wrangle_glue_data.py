"""
GLUE Dataset Wrangling for Encoder Training

This script extracts sentence pairs and NLI data from GLUE benchmark tasks
for use in contrastive learning. GLUE contains 9 diverse NLU tasks:

- CoLA: Grammatical acceptability (single sentences)
- SST-2: Sentiment analysis (single sentences)  
- MRPC: Paraphrase detection (sentence pairs)
- QQP: Quora question pairs (duplicate detection)
- STS-B: Semantic textual similarity (sentence pairs with scores)
- MNLI: Natural language inference (premise-hypothesis pairs)
- QNLI: Question-answering NLI (question-answer pairs)
- RTE: Recognizing textual entailment (premise-hypothesis)
- WNLI: Winograd NLI (coreference resolution)

The extracted data is formatted as triplets for contrastive learning.

Usage:
    python scripts/data_wrangling/wrangle_glue_data.py --tasks mrpc qqp stsb --output-dir data/glue/
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("Please install required packages: pip install datasets transformers")
    sys.exit(1)


class GLUEDataWrangler:
    """Extract training data from GLUE benchmark tasks."""
    
    # Task configurations
    TASK_CONFIGS = {
        'cola': {'type': 'single', 'num_labels': 2},
        'sst2': {'type': 'single', 'num_labels': 2},
        'mrpc': {'type': 'pair', 'num_labels': 2},
        'qqp': {'type': 'pair', 'num_labels': 2},
        'stsb': {'type': 'similarity', 'num_labels': 1},
        'mnli': {'type': 'nli', 'num_labels': 3},
        'qnli': {'type': 'qa_nli', 'num_labels': 2},
        'rte': {'type': 'nli', 'num_labels': 2},
        'wnli': {'type': 'nli', 'num_labels': 2}
    }
    
    def __init__(self, output_dir: str = "data/glue/"):
        """
        Initialize GLUE data wrangler.
        
        Args:
            output_dir: Directory to save extracted data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for different data types
        self.paraphrase_pairs = []
        self.similarity_pairs = []
        self.nli_triplets = []
        self.qa_pairs = []
        self.all_sentences = set()  # For negative sampling
        
    def extract_mrpc_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Extract paraphrase pairs from MRPC (Microsoft Research Paraphrase Corpus).
        
        Returns:
            List of (sentence1, sentence2, label) tuples where label=1 for paraphrases
        """
        print("Extracting MRPC paraphrase pairs...")
        pairs = []
        
        try:
            dataset = load_dataset('glue', 'mrpc', split='train')
            
            for sample in dataset:
                sent1 = sample['sentence1']
                sent2 = sample['sentence2']
                label = sample['label']  # 1 = paraphrase, 0 = not paraphrase
                
                pairs.append((sent1, sent2, label))
                self.all_sentences.add(sent1)
                self.all_sentences.add(sent2)
                
        except Exception as e:
            print(f"  Error loading MRPC: {e}")
            
        print(f"  Extracted {len(pairs)} pairs from MRPC")
        return pairs
    
    def extract_qqp_pairs(self, max_pairs: int = 50000) -> List[Tuple[str, str, int]]:
        """
        Extract question pairs from QQP (Quora Question Pairs).
        
        Args:
            max_pairs: Maximum number of pairs to extract
            
        Returns:
            List of (question1, question2, label) tuples where label=1 for duplicates
        """
        print("Extracting QQP question pairs...")
        pairs = []
        
        try:
            dataset = load_dataset('glue', 'qqp', split='train')
            
            for i, sample in enumerate(dataset):
                if i >= max_pairs:
                    break
                    
                q1 = sample['question1']
                q2 = sample['question2']
                label = sample['label']  # 1 = duplicate, 0 = not duplicate
                
                if q1 and q2:  # Some entries might be None
                    pairs.append((q1, q2, label))
                    self.all_sentences.add(q1)
                    self.all_sentences.add(q2)
                    
        except Exception as e:
            print(f"  Error loading QQP: {e}")
            
        print(f"  Extracted {len(pairs)} pairs from QQP")
        return pairs
    
    def extract_stsb_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Extract sentence pairs with similarity scores from STS-B.
        
        Returns:
            List of (sentence1, sentence2, score) tuples where score is 0-5
        """
        print("Extracting STS-B similarity pairs...")
        pairs = []
        
        try:
            dataset = load_dataset('glue', 'stsb', split='train')
            
            for sample in dataset:
                sent1 = sample['sentence1']
                sent2 = sample['sentence2']
                score = sample['label']  # Similarity score 0-5
                
                pairs.append((sent1, sent2, score))
                self.all_sentences.add(sent1)
                self.all_sentences.add(sent2)
                
        except Exception as e:
            print(f"  Error loading STS-B: {e}")
            
        print(f"  Extracted {len(pairs)} pairs from STS-B")
        return pairs
    
    def extract_mnli_triplets(self, max_samples: int = 50000) -> List[Tuple[str, str, str, int]]:
        """
        Extract NLI triplets from MNLI (Multi-Genre NLI).
        
        Args:
            max_samples: Maximum number of samples to extract
            
        Returns:
            List of (premise, hypothesis, label, genre) tuples
            where label: 0=entailment, 1=neutral, 2=contradiction
        """
        print("Extracting MNLI NLI triplets...")
        samples = []
        
        try:
            dataset = load_dataset('glue', 'mnli', split='train')
            
            for i, sample in enumerate(dataset):
                if i >= max_samples:
                    break
                    
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                
                samples.append((premise, hypothesis, label))
                self.all_sentences.add(premise)
                self.all_sentences.add(hypothesis)
                
        except Exception as e:
            print(f"  Error loading MNLI: {e}")
            
        print(f"  Extracted {len(samples)} NLI samples from MNLI")
        return samples
    
    def extract_qnli_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Extract question-answer pairs from QNLI.
        
        Returns:
            List of (question, sentence, label) tuples
            where label=1 if sentence answers the question
        """
        print("Extracting QNLI question-answer pairs...")
        pairs = []
        
        try:
            dataset = load_dataset('glue', 'qnli', split='train')
            
            for sample in dataset:
                question = sample['question']
                sentence = sample['sentence']
                label = sample['label']  # 0 = entailment (answers), 1 = not entailment
                
                # Invert label so 1 = positive pair
                pairs.append((question, sentence, 1 - label))
                self.all_sentences.add(question)
                self.all_sentences.add(sentence)
                
        except Exception as e:
            print(f"  Error loading QNLI: {e}")
            
        print(f"  Extracted {len(pairs)} pairs from QNLI")
        return pairs
    
    def pairs_to_triplets(self, pairs: List[Tuple[str, str, int]], 
                         sample_ratio: float = 1.0) -> List[Tuple[str, str, str]]:
        """
        Convert binary pairs to triplets for contrastive learning.
        
        Args:
            pairs: List of (sent1, sent2, label) tuples
            sample_ratio: Fraction of pairs to convert
            
        Returns:
            List of (anchor, positive, negative) triplets
        """
        triplets = []
        
        # Separate positive and negative pairs
        positive_pairs = [(s1, s2) for s1, s2, label in pairs if label == 1]
        negative_pool = [(s1, s2) for s1, s2, label in pairs if label == 0]
        
        # If no explicit negatives, use random sentences
        if not negative_pool:
            negative_pool = list(self.all_sentences)
        
        # Sample subset if requested
        if sample_ratio < 1.0:
            n_samples = int(len(positive_pairs) * sample_ratio)
            positive_pairs = random.sample(positive_pairs, min(n_samples, len(positive_pairs)))
        
        # Generate triplets
        for sent1, sent2 in positive_pairs:
            # For negative, either use a negative pair or random sentence
            if negative_pool and isinstance(negative_pool[0], tuple):
                # Use second sentence from a negative pair
                neg_pair = random.choice(negative_pool)
                negative = neg_pair[1] if neg_pair[0] != sent1 else neg_pair[0]
            else:
                # Use random sentence different from positive
                candidates = [s for s in negative_pool if s != sent2 and s != sent1]
                if candidates:
                    negative = random.choice(candidates)
                else:
                    continue
                    
            triplets.append((sent1, sent2, negative))
            
        return triplets
    
    def nli_to_triplets(self, nli_samples: List[Tuple[str, str, int]]) -> List[Tuple[str, str, str]]:
        """
        Convert NLI samples to triplets.
        
        For NLI:
        - Entailment pairs → positive pairs
        - Contradiction pairs → negative pairs
        - Neutral pairs → can be used as harder negatives
        
        Args:
            nli_samples: List of (premise, hypothesis, label) tuples
            
        Returns:
            List of (anchor, positive, negative) triplets
        """
        triplets = []
        
        # Group by premise
        premise_groups = defaultdict(list)
        for premise, hypothesis, label in nli_samples:
            premise_groups[premise].append((hypothesis, label))
        
        # Generate triplets from each premise group
        for premise, hypotheses in premise_groups.items():
            entailments = [h for h, l in hypotheses if l == 0]
            contradictions = [h for h, l in hypotheses if l == 2]
            neutrals = [h for h, l in hypotheses if l == 1]
            
            # Create triplets: premise + entailment (positive) + contradiction (negative)
            for entailment in entailments:
                if contradictions:
                    negative = random.choice(contradictions)
                elif neutrals:
                    negative = random.choice(neutrals)
                else:
                    continue
                    
                triplets.append((premise, entailment, negative))
                
            # Also create triplets between entailment pairs if multiple exist
            if len(entailments) > 1:
                for i, ent1 in enumerate(entailments):
                    for ent2 in entailments[i+1:]:
                        if contradictions:
                            negative = random.choice(contradictions)
                        elif neutrals:
                            negative = random.choice(neutrals)
                        else:
                            continue
                        triplets.append((ent1, ent2, negative))
                        
        return triplets
    
    def similarity_to_triplets(self, pairs: List[Tuple[str, str, float]], 
                             pos_threshold: float = 4.0,
                             neg_threshold: float = 2.0) -> List[Tuple[str, str, str]]:
        """
        Convert similarity pairs to triplets.
        
        Args:
            pairs: List of (sent1, sent2, score) tuples (score 0-5)
            pos_threshold: Minimum score for positive pairs
            neg_threshold: Maximum score for negative pairs
            
        Returns:
            List of (anchor, positive, negative) triplets
        """
        triplets = []
        
        # Separate by similarity
        positive_pairs = [(s1, s2) for s1, s2, score in pairs if score >= pos_threshold]
        negative_candidates = [s2 for s1, s2, score in pairs if score <= neg_threshold]
        
        # If not enough negatives, use all low-similarity sentences
        if len(negative_candidates) < len(positive_pairs):
            negative_candidates.extend([s1 for s1, s2, score in pairs if score <= neg_threshold])
        
        # Generate triplets
        for sent1, sent2 in positive_pairs:
            candidates = [n for n in negative_candidates if n != sent2 and n != sent1]
            if candidates:
                negative = random.choice(candidates)
                triplets.append((sent1, sent2, negative))
                
        return triplets
    
    def augment_with_paraphrasing(self, triplets: List[Tuple[str, str, str]], 
                                 augment_ratio: float = 0.2) -> List[Tuple[str, str, str]]:
        """
        Augment triplets with simple paraphrasing techniques.
        
        Args:
            triplets: Original triplets
            augment_ratio: Fraction of triplets to augment
            
        Returns:
            Extended list of triplets
        """
        augmented = []
        n_augment = int(len(triplets) * augment_ratio)
        
        for i in range(n_augment):
            anchor, positive, negative = random.choice(triplets)
            
            # Simple augmentation techniques
            augment_type = random.choice(['shuffle', 'truncate', 'capitalize'])
            
            if augment_type == 'shuffle':
                # Shuffle words in positive
                words = positive.split()
                random.shuffle(words)
                new_positive = ' '.join(words)
                
            elif augment_type == 'truncate':
                # Use first/last part of sentences
                words = positive.split()
                if len(words) > 5:
                    if random.random() < 0.5:
                        new_positive = ' '.join(words[:len(words)//2])
                    else:
                        new_positive = ' '.join(words[len(words)//2:])
                else:
                    new_positive = positive
                    
            else:  # capitalize
                # Change capitalization
                if random.random() < 0.5:
                    new_positive = positive.lower()
                else:
                    new_positive = positive.upper()
                    
            augmented.append((anchor, new_positive, negative))
            
        return triplets + augmented
    
    def wrangle_all_tasks(self, task_names: List[str] = None):
        """
        Extract data from specified GLUE tasks.
        
        Args:
            task_names: List of GLUE task names to process
        """
        if task_names is None:
            task_names = ['mrpc', 'qqp', 'stsb', 'mnli', 'qnli']
        
        all_triplets = []
        metadata = {
            'tasks_processed': [],
            'total_samples': 0,
            'data_types': defaultdict(int)
        }
        
        print("\n" + "="*80)
        print("GLUE Data Extraction")
        print("="*80)
        
        # Extract paraphrase pairs (MRPC)
        if 'mrpc' in task_names:
            pairs = self.extract_mrpc_pairs()
            self.paraphrase_pairs.extend(pairs)
            triplets = self.pairs_to_triplets(pairs)
            all_triplets.extend(triplets)
            metadata['tasks_processed'].append('mrpc')
            metadata['data_types']['paraphrase'] += len(triplets)
        
        # Extract question pairs (QQP)
        if 'qqp' in task_names:
            pairs = self.extract_qqp_pairs(max_pairs=20000)
            self.paraphrase_pairs.extend(pairs)
            triplets = self.pairs_to_triplets(pairs)
            all_triplets.extend(triplets)
            metadata['tasks_processed'].append('qqp')
            metadata['data_types']['question_pairs'] += len(triplets)
        
        # Extract similarity pairs (STS-B)
        if 'stsb' in task_names:
            pairs = self.extract_stsb_pairs()
            self.similarity_pairs.extend(pairs)
            triplets = self.similarity_to_triplets(pairs)
            all_triplets.extend(triplets)
            metadata['tasks_processed'].append('stsb')
            metadata['data_types']['similarity'] += len(triplets)
        
        # Extract NLI data (MNLI)
        if 'mnli' in task_names:
            nli_samples = self.extract_mnli_triplets(max_samples=20000)
            triplets = self.nli_to_triplets(nli_samples)
            self.nli_triplets.extend(triplets)
            all_triplets.extend(triplets)
            metadata['tasks_processed'].append('mnli')
            metadata['data_types']['nli'] += len(triplets)
        
        # Extract QA pairs (QNLI)
        if 'qnli' in task_names:
            pairs = self.extract_qnli_pairs()
            self.qa_pairs.extend(pairs)
            triplets = self.pairs_to_triplets(pairs, sample_ratio=0.5)
            all_triplets.extend(triplets)
            metadata['tasks_processed'].append('qnli')
            metadata['data_types']['qa'] += len(triplets)
        
        # Augment data
        print("\nAugmenting data with paraphrasing...")
        all_triplets = self.augment_with_paraphrasing(all_triplets, augment_ratio=0.1)
        
        # Update metadata
        metadata['total_samples'] = len(all_triplets)
        metadata['unique_sentences'] = len(self.all_sentences)
        
        # Save data
        self.save_data(all_triplets, metadata)
        
        return all_triplets, metadata
    
    def save_data(self, triplets: List[Tuple[str, str, str]], metadata: Dict):
        """
        Save extracted data to disk.
        
        Args:
            triplets: List of (anchor, positive, negative) tuples
            metadata: Metadata about extraction
        """
        print("\n" + "="*80)
        print("Saving Data")
        print("="*80)
        
        # Save triplets
        triplets_file = self.output_dir / "glue_triplets.pkl"
        with open(triplets_file, 'wb') as f:
            pickle.dump(triplets, f)
        print(f"Saved {len(triplets)} triplets to {triplets_file}")
        
        # Save sample as JSON
        sample_file = self.output_dir / "glue_samples.json"
        sample_data = {
            'metadata': dict(metadata),
            'sample_triplets': [
                {'anchor': a, 'positive': p, 'negative': n}
                for a, p, n in triplets[:100]
            ]
        }
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Saved sample data to {sample_file}")
        
        # Save component data
        if self.paraphrase_pairs:
            para_file = self.output_dir / "glue_paraphrase_pairs.pkl"
            with open(para_file, 'wb') as f:
                pickle.dump(self.paraphrase_pairs, f)
            print(f"Saved {len(self.paraphrase_pairs)} paraphrase pairs")
        
        if self.similarity_pairs:
            sim_file = self.output_dir / "glue_similarity_pairs.pkl"
            with open(sim_file, 'wb') as f:
                pickle.dump(self.similarity_pairs, f)
            print(f"Saved {len(self.similarity_pairs)} similarity pairs")
        
        # Save metadata
        metadata_file = self.output_dir / "glue_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(dict(metadata), f, indent=2)
        print(f"Saved metadata to {metadata_file}")
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print(f"Total triplets: {len(triplets)}")
        print(f"Unique sentences: {metadata['unique_sentences']}")
        print(f"Tasks processed: {metadata['tasks_processed']}")
        print("\nData type breakdown:")
        for dtype, count in metadata['data_types'].items():
            print(f"  {dtype}: {count}")
        print(f"\nOutput directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract training data from GLUE benchmark tasks"
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['mrpc', 'qqp', 'stsb', 'mnli', 'qnli'],
        choices=['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli'],
        help='GLUE tasks to extract data from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/glue/',
        help='Directory to save extracted data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Run extraction
    wrangler = GLUEDataWrangler(output_dir=args.output_dir)
    triplets, metadata = wrangler.wrangle_all_tasks(task_names=args.tasks)
    
    print("\nData wrangling complete!")
    print(f"Use the extracted data for training with:")
    print(f"  python src/Training/train_with_benchmarks.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()