"""
MTEB Dataset Wrangling for Encoder Training

This script extracts high-quality text pairs and triplets from MTEB benchmark datasets
for use in contrastive learning. MTEB contains 56+ tasks across 8 categories:
- Semantic Textual Similarity (STS)
- Retrieval (query-document pairs)
- Clustering
- Pair Classification
- Reranking
- Classification
- Summarization
- BitextMining (parallel sentences)

The extracted data is formatted for contrastive learning with positive and negative pairs.

Usage:
    python scripts/data_wrangling/wrangle_mteb_data.py --tasks sts retrieval --output-dir data/mteb/
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

# MTEB imports
try:
    from mteb import MTEB
    from datasets import load_dataset
except ImportError:
    print("Please install MTEB: pip install mteb datasets")
    sys.exit(1)


class MTEBDataWrangler:
    """Extract training data from MTEB benchmark tasks."""
    
    # High-quality STS datasets for similarity pairs
    STS_TASKS = [
        "STS12", "STS13", "STS14", "STS15", "STS16",
        "STS17", "STS22", "STSBenchmark", "BIOSSES", "SICK-R"
    ]
    
    # Retrieval tasks for query-document pairs
    RETRIEVAL_TASKS = [
        "MSMARCOv2", "NQ", "HotpotQA", "FiQA2018", 
        "SciFact", "NFCorpus", "ArguAna", "Touche2020",
        "CQADupstackRetrieval", "Quora", "DBPedia", "SCIDOCS",
        "FEVER", "Climate-FEVER", "NaturalQuestions"
    ]
    
    # Clustering tasks for semantic groups
    CLUSTERING_TASKS = [
        "RedditClustering", "StackExchangeClustering",
        "TwentyNewsgroupsClustering", "MedrxivClusteringS2S",
        "ArxivClusteringS2S", "BiorxivClusteringS2S"
    ]
    
    # Pair classification for positive/negative pairs
    PAIR_CLASSIFICATION_TASKS = [
        "TwitterSemEval2015", "TwitterURLCorpus",
        "SprintDuplicateQuestions", "AmazonCounterfactualClassification"
    ]
    
    def __init__(self, output_dir: str = "data/mteb/"):
        """
        Initialize MTEB data wrangler.
        
        Args:
            output_dir: Directory to save extracted data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sts_data = []
        self.retrieval_data = []
        self.clustering_data = []
        self.pair_data = []
        
    def extract_sts_pairs(self, task_name: str) -> List[Tuple[str, str, float]]:
        """
        Extract sentence pairs with similarity scores from STS tasks.
        
        Args:
            task_name: Name of STS task
            
        Returns:
            List of (sentence1, sentence2, similarity_score) tuples
        """
        print(f"Extracting STS pairs from {task_name}...")
        pairs = []
        
        try:
            # Load dataset
            if task_name == "STSBenchmark":
                dataset = load_dataset("sentence-transformers/stsb", split="train")
            elif task_name.startswith("STS"):
                # Load STS12-22 datasets
                year = task_name[3:]
                dataset = load_dataset(f"mteb/sts{year}", split="train")
            else:
                dataset = load_dataset(f"mteb/{task_name.lower()}", split="train")
            
            # Extract pairs
            for sample in dataset:
                sent1 = sample.get("sentence1", sample.get("text1", ""))
                sent2 = sample.get("sentence2", sample.get("text2", ""))
                score = sample.get("score", sample.get("similarity_score", 0.0))
                
                if sent1 and sent2:
                    # Normalize score to [0, 1]
                    if score > 1:
                        score = score / 5.0  # STS scores are 0-5
                    
                    pairs.append((sent1, sent2, score))
                    
        except Exception as e:
            print(f"  Error loading {task_name}: {e}")
            
        print(f"  Extracted {len(pairs)} pairs from {task_name}")
        return pairs
    
    def extract_retrieval_pairs(self, task_name: str, max_pairs: int = 10000) -> List[Tuple[str, str, str]]:
        """
        Extract query-document pairs from retrieval tasks.
        
        Args:
            task_name: Name of retrieval task
            max_pairs: Maximum number of pairs to extract
            
        Returns:
            List of (query, positive_doc, negative_doc) tuples
        """
        print(f"Extracting retrieval pairs from {task_name}...")
        triplets = []
        
        try:
            # Load dataset (simplified - actual loading varies by task)
            dataset = load_dataset(f"mteb/{task_name.lower()}", split="train")
            
            for i, sample in enumerate(dataset):
                if i >= max_pairs:
                    break
                    
                query = sample.get("query", "")
                positive = sample.get("positive", sample.get("doc", ""))
                negatives = sample.get("negatives", [])
                
                if query and positive and negatives:
                    # Create triplet with random negative
                    negative = random.choice(negatives) if isinstance(negatives, list) else negatives
                    triplets.append((query, positive, negative))
                    
        except Exception as e:
            print(f"  Error loading {task_name}: {e}")
            # Try alternative loading method for some datasets
            try:
                if task_name == "MSMARCOv2":
                    # Special handling for MSMARCO
                    triplets = self._extract_msmarco_pairs(max_pairs)
                elif task_name == "NaturalQuestions":
                    # Special handling for NQ
                    triplets = self._extract_nq_pairs(max_pairs)
            except:
                pass
                
        print(f"  Extracted {len(triplets)} triplets from {task_name}")
        return triplets
    
    def _extract_msmarco_pairs(self, max_pairs: int) -> List[Tuple[str, str, str]]:
        """Extract pairs from MSMARCO dataset."""
        triplets = []
        try:
            dataset = load_dataset("sentence-transformers/msmarco-hard-negatives", split="train")
            
            for i, sample in enumerate(dataset):
                if i >= max_pairs:
                    break
                    
                query = sample["query"]
                positive = sample["positive"]
                negative = sample["negative"][0] if sample["negative"] else ""
                
                if query and positive and negative:
                    triplets.append((query, positive, negative))
                    
        except Exception as e:
            print(f"    Could not load MSMARCO: {e}")
            
        return triplets
    
    def _extract_nq_pairs(self, max_pairs: int) -> List[Tuple[str, str, str]]:
        """Extract pairs from Natural Questions dataset."""
        triplets = []
        try:
            dataset = load_dataset("natural_questions", "default", split="train")
            
            for i, sample in enumerate(dataset):
                if i >= max_pairs:
                    break
                    
                question = sample["question_text"]
                # Extract long answer as positive
                if sample["annotations"]["long_answer"]:
                    start = sample["annotations"]["long_answer"][0]["start_byte"]
                    end = sample["annotations"]["long_answer"][0]["end_byte"]
                    positive = sample["document_text"][start:end]
                    
                    # Use random other part as negative
                    doc_parts = sample["document_text"].split("\n")
                    negative = random.choice([p for p in doc_parts if p != positive and len(p) > 50])
                    
                    if question and positive and negative:
                        triplets.append((question, positive[:500], negative[:500]))  # Truncate
                        
        except Exception as e:
            print(f"    Could not load NQ: {e}")
            
        return triplets
    
    def generate_triplets_from_pairs(self, pairs: List[Tuple[str, str, float]], 
                                    threshold: float = 0.7) -> List[Tuple[str, str, str]]:
        """
        Generate triplets from similarity pairs.
        
        Args:
            pairs: List of (sent1, sent2, similarity) tuples
            threshold: Similarity threshold for positive pairs
            
        Returns:
            List of (anchor, positive, negative) triplets
        """
        triplets = []
        
        # Separate into positive and negative pairs
        positive_pairs = [(s1, s2) for s1, s2, score in pairs if score >= threshold]
        negative_candidates = [s2 for s1, s2, score in pairs if score < 0.3]
        
        if not negative_candidates:
            # Use all second sentences as potential negatives
            negative_candidates = [s2 for _, s2, _ in pairs]
        
        # Generate triplets
        for sent1, sent2 in positive_pairs:
            # Select random negative that's different from positive
            negatives = [n for n in negative_candidates if n != sent2]
            if negatives:
                negative = random.choice(negatives)
                triplets.append((sent1, sent2, negative))
                
        return triplets
    
    def extract_clustering_groups(self, task_name: str) -> Dict[str, List[str]]:
        """
        Extract semantic groups from clustering tasks.
        
        Args:
            task_name: Name of clustering task
            
        Returns:
            Dict mapping cluster_id to list of texts
        """
        print(f"Extracting clustering groups from {task_name}...")
        clusters = defaultdict(list)
        
        try:
            dataset = load_dataset(f"mteb/{task_name.lower()}", split="test")
            
            for sample in dataset:
                text = sample.get("text", sample.get("sentence", ""))
                label = sample.get("label", sample.get("cluster", -1))
                
                if text and label >= 0:
                    clusters[str(label)].append(text)
                    
        except Exception as e:
            print(f"  Error loading {task_name}: {e}")
            
        print(f"  Extracted {len(clusters)} clusters from {task_name}")
        return dict(clusters)
    
    def clusters_to_triplets(self, clusters: Dict[str, List[str]], 
                           samples_per_cluster: int = 100) -> List[Tuple[str, str, str]]:
        """
        Convert clusters to triplets for contrastive learning.
        
        Args:
            clusters: Dict mapping cluster_id to texts
            samples_per_cluster: Number of triplets per cluster
            
        Returns:
            List of (anchor, positive, negative) triplets
        """
        triplets = []
        cluster_ids = list(clusters.keys())
        
        for cluster_id in cluster_ids:
            texts = clusters[cluster_id]
            if len(texts) < 2:
                continue
                
            # Generate within-cluster pairs as positives
            for _ in range(min(samples_per_cluster, len(texts) * 2)):
                anchor = random.choice(texts)
                positive = random.choice([t for t in texts if t != anchor])
                
                # Select negative from different cluster
                other_cluster = random.choice([c for c in cluster_ids if c != cluster_id])
                negative = random.choice(clusters[other_cluster])
                
                triplets.append((anchor, positive, negative))
                
        return triplets
    
    def wrangle_all_tasks(self, task_categories: List[str] = None):
        """
        Extract data from all specified task categories.
        
        Args:
            task_categories: List of task categories to process
                           Options: ['sts', 'retrieval', 'clustering', 'pair_classification']
        """
        if task_categories is None:
            task_categories = ['sts', 'retrieval']
        
        all_triplets = []
        metadata = {
            'task_categories': task_categories,
            'tasks_processed': [],
            'total_samples': 0
        }
        
        # Extract STS data
        if 'sts' in task_categories:
            print("\n" + "="*80)
            print("Extracting STS Tasks")
            print("="*80)
            
            for task in self.STS_TASKS[:3]:  # Start with first 3 for testing
                pairs = self.extract_sts_pairs(task)
                if pairs:
                    self.sts_data.extend(pairs)
                    triplets = self.generate_triplets_from_pairs(pairs)
                    all_triplets.extend(triplets)
                    metadata['tasks_processed'].append(task)
            
            print(f"\nTotal STS pairs: {len(self.sts_data)}")
            print(f"Total STS triplets: {len([t for t in all_triplets if True])}")
        
        # Extract retrieval data
        if 'retrieval' in task_categories:
            print("\n" + "="*80)
            print("Extracting Retrieval Tasks")
            print("="*80)
            
            for task in ["MSMARCOv2", "NaturalQuestions"]:  # Start with 2 major tasks
                triplets = self.extract_retrieval_pairs(task, max_pairs=5000)
                if triplets:
                    self.retrieval_data.extend(triplets)
                    all_triplets.extend(triplets)
                    metadata['tasks_processed'].append(task)
            
            print(f"\nTotal retrieval triplets: {len(self.retrieval_data)}")
        
        # Extract clustering data
        if 'clustering' in task_categories:
            print("\n" + "="*80)
            print("Extracting Clustering Tasks")
            print("="*80)
            
            for task in self.CLUSTERING_TASKS[:2]:  # Start with first 2
                clusters = self.extract_clustering_groups(task)
                if clusters:
                    triplets = self.clusters_to_triplets(clusters)
                    self.clustering_data.extend(triplets)
                    all_triplets.extend(triplets)
                    metadata['tasks_processed'].append(task)
            
            print(f"\nTotal clustering triplets: {len(self.clustering_data)}")
        
        # Save data
        metadata['total_samples'] = len(all_triplets)
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
        triplets_file = self.output_dir / "mteb_triplets.pkl"
        with open(triplets_file, 'wb') as f:
            pickle.dump(triplets, f)
        print(f"Saved {len(triplets)} triplets to {triplets_file}")
        
        # Save as JSON for readability (first 100 samples)
        sample_file = self.output_dir / "mteb_samples.json"
        sample_data = {
            'metadata': metadata,
            'sample_triplets': [
                {'anchor': a, 'positive': p, 'negative': n}
                for a, p, n in triplets[:100]
            ]
        }
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Saved sample data to {sample_file}")
        
        # Save individual components if available
        if self.sts_data:
            sts_file = self.output_dir / "mteb_sts_pairs.pkl"
            with open(sts_file, 'wb') as f:
                pickle.dump(self.sts_data, f)
            print(f"Saved {len(self.sts_data)} STS pairs to {sts_file}")
        
        # Save metadata
        metadata_file = self.output_dir / "mteb_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print(f"Total triplets extracted: {len(triplets)}")
        print(f"Tasks processed: {len(metadata['tasks_processed'])}")
        print(f"Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract training data from MTEB benchmark tasks"
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['sts', 'retrieval'],
        choices=['sts', 'retrieval', 'clustering', 'pair_classification'],
        help='Task categories to extract data from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/mteb/',
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
    wrangler = MTEBDataWrangler(output_dir=args.output_dir)
    triplets, metadata = wrangler.wrangle_all_tasks(task_categories=args.tasks)
    
    print("\nData wrangling complete!")
    print(f"Use the extracted data for training with:")
    print(f"  python src/Training/train_with_benchmarks.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()