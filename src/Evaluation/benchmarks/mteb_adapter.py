"""
MTEB (Massive Text Embedding Benchmark) Evaluation Adapter

This module provides an adapter to evaluate trained encoders on the MTEB benchmark,
which includes 56+ tasks across 8 categories:
1. Bitext Mining
2. Classification  
3. Clustering
4. Pair Classification
5. Reranking
6. Retrieval
7. STS (Semantic Textual Similarity)
8. Summarization

MTEB provides a standardized way to evaluate text embeddings across diverse tasks
and compare against state-of-the-art models on the leaderboard.

Usage:
    from src.Evaluation.benchmarks.mteb_adapter import MTEBEvaluator
    from src.Encoders import Text_to_Vec
    
    encoder = Text_to_Vec()
    evaluator = MTEBEvaluator(encoder)
    
    # Run on specific tasks
    results = evaluator.run(tasks=['STS12', 'STS13', 'Banking77Classification'])
    
    # Run on all STS tasks
    results = evaluator.run_category('STS')
    
    # Generate leaderboard submission
    evaluator.create_submission(results, 'results/mteb_submission.json')
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import torch
import torch.nn as nn
from tqdm import tqdm

# MTEB imports
try:
    from mteb import MTEB
    from sentence_transformers import SentenceTransformer
    MTEB_AVAILABLE = True
except ImportError:
    MTEB_AVAILABLE = False
    print("Warning: MTEB not installed. Install with: pip install mteb sentence-transformers")


class EncoderWrapper(nn.Module):
    """
    Wrapper to make our encoders compatible with MTEB/sentence-transformers interface.
    
    MTEB expects a model with an `encode()` method that takes a list of strings
    and returns numpy arrays of embeddings.
    """
    
    def __init__(self, encoder, max_length: int = 512, batch_size: int = 32):
        """
        Initialize encoder wrapper.
        
        Args:
            encoder: Our Text_to_Vec or similar encoder
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
        """
        super().__init__()
        self.encoder = encoder
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = next(encoder.parameters()).device
        
    def encode(
        self,
        sentences: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences to embeddings (MTEB-compatible interface).
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress
            convert_to_numpy: Convert to numpy (required by MTEB)
            convert_to_tensor: Keep as tensor
            normalize_embeddings: L2 normalize embeddings
            
        Returns:
            Embeddings as numpy array or tensor
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(sentences) + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")
        
        with torch.no_grad():
            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(sentences))
                batch_sentences = sentences[start_idx:end_idx]
                
                # Encode using our encoder
                embeddings = self.encoder(batch_sentences, device=self.device)
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Convert to appropriate format
        if convert_to_numpy:
            return all_embeddings.cpu().numpy()
        else:
            return all_embeddings


class MTEBEvaluator:
    """
    Evaluator for MTEB benchmark tasks.
    
    Handles evaluation on various task types and provides
    formatted results for comparison with other models.
    """
    
    # Task categories and their default tasks
    TASK_CATEGORIES = {
        'STS': [
            'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
            'STS17', 'STS22', 'STSBenchmark', 'BIOSSES', 'SICK-R'
        ],
        'Classification': [
            'AmazonCounterfactualClassification', 'AmazonPolarityClassification',
            'AmazonReviewsClassification', 'Banking77Classification',
            'EmotionClassification', 'ImdbClassification', 'MassiveIntentClassification',
            'MassiveScenarioClassification', 'MTOPDomainClassification',
            'MTOPIntentClassification', 'ToxicConversationsClassification',
            'TweetSentimentExtractionClassification'
        ],
        'Clustering': [
            'ArxivClusteringP2P', 'ArxivClusteringS2S', 'BiorxivClusteringP2P',
            'BiorxivClusteringS2S', 'MedrxivClusteringP2P', 'MedrxivClusteringS2S',
            'RedditClustering', 'RedditClusteringP2P', 'StackExchangeClustering',
            'StackExchangeClusteringP2P', 'TwentyNewsgroupsClustering'
        ],
        'PairClassification': [
            'SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'
        ],
        'Reranking': [
            'AskUbuntuDupQuestions', 'MindSmallReranking', 'SciDocsRR', 
            'StackOverflowDupQuestions'
        ],
        'Retrieval': [
            'ArguAna', 'ClimateFEVER', 'CQADupstackRetrieval', 'DBPedia',
            'FEVER', 'FiQA2018', 'HotpotQA', 'MSMARCO', 'MSMARCOv2',
            'NFCorpus', 'NQ', 'QuoraRetrieval', 'SCIDOCS', 'SciFact',
            'Touche2020', 'TRECCOVID'
        ],
        'BitextMining': [
            'BUCC', 'Tatoeba'
        ],
        'Summarization': [
            'SummEval'
        ]
    }
    
    # Quick evaluation subset (for testing)
    QUICK_TASKS = [
        'STS12', 'Banking77Classification', 'TwentyNewsgroupsClustering',
        'SprintDuplicateQuestions', 'SciFact'
    ]
    
    def __init__(
        self,
        encoder,
        batch_size: int = 32,
        max_length: int = 512,
        device: str = None
    ):
        """
        Initialize MTEB evaluator.
        
        Args:
            encoder: Text encoder to evaluate (Text_to_Vec or similar)
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            device: Device for computation
        """
        if not MTEB_AVAILABLE:
            raise ImportError("MTEB not installed. Run: pip install mteb sentence-transformers")
        
        self.encoder = encoder
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Move encoder to device
        self.encoder.to(device)
        self.encoder.eval()
        
        # Create wrapped model for MTEB compatibility
        self.wrapped_model = EncoderWrapper(encoder, max_length, batch_size)
        
    def run(
        self,
        tasks: Optional[List[str]] = None,
        output_folder: str = 'results/mteb/',
        eval_splits: List[str] = ['test'],
        save_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Run evaluation on specified MTEB tasks.
        
        Args:
            tasks: List of task names (None = use quick subset)
            output_folder: Folder to save results
            eval_splits: Which splits to evaluate ('train', 'dev', 'test')
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary of results by task
        """
        if tasks is None:
            tasks = self.QUICK_TASKS
            print(f"Running quick evaluation on {len(tasks)} tasks")
        else:
            print(f"Running evaluation on {len(tasks)} tasks")
        
        # Create output folder
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MTEB
        evaluation = MTEB(tasks=tasks)
        
        # Run evaluation
        print("\nStarting MTEB evaluation...")
        print("="*80)
        
        results = evaluation.run(
            self.wrapped_model,
            output_folder=str(output_path),
            eval_splits=eval_splits,
            save_predictions=save_predictions,
            batch_size=self.batch_size
        )
        
        # Process and format results
        formatted_results = self._format_results(results, tasks)
        
        # Save formatted results
        results_file = output_path / 'mteb_results.json'
        with open(results_file, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return formatted_results
    
    def run_category(
        self,
        category: str,
        output_folder: str = 'results/mteb/',
        eval_splits: List[str] = ['test']
    ) -> Dict[str, Any]:
        """
        Run evaluation on all tasks in a category.
        
        Args:
            category: Category name ('STS', 'Classification', etc.)
            output_folder: Folder to save results
            eval_splits: Which splits to evaluate
            
        Returns:
            Dictionary of results
        """
        if category not in self.TASK_CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Choose from {list(self.TASK_CATEGORIES.keys())}")
        
        tasks = self.TASK_CATEGORIES[category]
        print(f"Running {category} evaluation ({len(tasks)} tasks)")
        
        return self.run(tasks, output_folder, eval_splits)
    
    def _format_results(self, raw_results: List[Dict], task_names: List[str]) -> Dict[str, Any]:
        """
        Format MTEB results for analysis and comparison.
        
        Args:
            raw_results: Raw results from MTEB
            task_names: List of task names
            
        Returns:
            Formatted results dictionary
        """
        formatted = {
            'task_results': {},
            'category_averages': defaultdict(list),
            'overall_score': 0.0,
            'metadata': {
                'num_tasks': len(task_names),
                'tasks': task_names
            }
        }
        
        all_scores = []
        
        for result in raw_results:
            task_name = result.get('task_name', result.get('dataset', 'unknown'))
            
            # Extract main score based on task type
            if 'test' in result:
                test_results = result['test']
                
                # Different tasks have different primary metrics
                if 'map' in test_results:  # Retrieval
                    score = test_results['map']
                    metric = 'MAP'
                elif 'accuracy' in test_results:  # Classification
                    score = test_results['accuracy']
                    metric = 'Accuracy'
                elif 'ap' in test_results:  # Reranking
                    score = test_results['ap']
                    metric = 'AP'
                elif 'v_measure' in test_results:  # Clustering
                    score = test_results['v_measure']
                    metric = 'V-Measure'
                elif 'cos_sim' in test_results:  # STS
                    if 'spearman' in test_results['cos_sim']:
                        score = test_results['cos_sim']['spearman']
                        metric = 'Spearman'
                    else:
                        score = test_results['cos_sim'].get('pearson', 0.0)
                        metric = 'Pearson'
                elif 'f1' in test_results:  # Pair Classification
                    score = test_results['f1']
                    metric = 'F1'
                else:
                    # Try to find any score
                    score = 0.0
                    metric = 'Unknown'
                    for key in ['accuracy', 'f1', 'map', 'spearman']:
                        if key in test_results:
                            score = test_results[key]
                            metric = key.capitalize()
                            break
                
                # Store results
                formatted['task_results'][task_name] = {
                    'score': score,
                    'metric': metric,
                    'full_results': test_results
                }
                
                all_scores.append(score)
                
                # Categorize task
                category = self._get_task_category(task_name)
                formatted['category_averages'][category].append(score)
        
        # Calculate averages
        if all_scores:
            formatted['overall_score'] = np.mean(all_scores)
        
        # Calculate category averages
        for category in formatted['category_averages']:
            scores = formatted['category_averages'][category]
            formatted['category_averages'][category] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
        
        return formatted
    
    def _get_task_category(self, task_name: str) -> str:
        """Get category for a task name."""
        for category, tasks in self.TASK_CATEGORIES.items():
            if task_name in tasks:
                return category
        
        # Try to infer from task name
        task_lower = task_name.lower()
        if 'sts' in task_lower:
            return 'STS'
        elif 'classification' in task_lower:
            return 'Classification'
        elif 'clustering' in task_lower:
            return 'Clustering'
        elif 'retrieval' in task_lower or 'msmarco' in task_lower:
            return 'Retrieval'
        elif 'rerank' in task_lower:
            return 'Reranking'
        else:
            return 'Other'
    
    def create_submission(
        self,
        results: Dict[str, Any],
        output_file: str = 'mteb_submission.json',
        model_name: str = 'SemanticVectorSpace',
        model_revision: str = 'v1.0'
    ):
        """
        Create MTEB leaderboard submission file.
        
        Args:
            results: Results from run() or run_category()
            output_file: Path to save submission
            model_name: Name of your model
            model_revision: Model version/revision
        """
        submission = {
            'model_name': model_name,
            'model_revision': model_revision,
            'results': {}
        }
        
        # Format results for submission
        for task_name, task_results in results['task_results'].items():
            submission['results'][task_name] = {
                'test': task_results['full_results']
            }
        
        # Add metadata
        submission['metadata'] = {
            'overall_score': results['overall_score'],
            'num_tasks': results['metadata']['num_tasks'],
            'category_scores': dict(results['category_averages'])
        }
        
        # Save submission
        with open(output_file, 'w') as f:
            json.dump(submission, f, indent=2)
        
        print(f"Submission file created: {output_file}")
        print(f"Overall score: {results['overall_score']:.3f}")
        print("\nCategory scores:")
        for category, scores in results['category_averages'].items():
            if isinstance(scores, dict):
                print(f"  {category}: {scores['mean']:.3f} ± {scores['std']:.3f}")
    
    def compare_with_baseline(
        self,
        results: Dict[str, Any],
        baseline: str = 'all-MiniLM-L6-v2'
    ) -> Dict[str, float]:
        """
        Compare results with a baseline model.
        
        Args:
            results: Your model's results
            baseline: Baseline model name (must be on MTEB leaderboard)
            
        Returns:
            Comparison statistics
        """
        # Load baseline results (would need to be downloaded/cached)
        # For now, we'll use approximate scores for common models
        BASELINE_SCORES = {
            'all-MiniLM-L6-v2': 0.630,
            'all-mpnet-base-v2': 0.697,
            'e5-large-v2': 0.754,
            'gte-large': 0.763,
            'bge-large-en-v1.5': 0.768
        }
        
        if baseline not in BASELINE_SCORES:
            print(f"Baseline {baseline} not found. Available: {list(BASELINE_SCORES.keys())}")
            return {}
        
        baseline_score = BASELINE_SCORES[baseline]
        your_score = results['overall_score']
        
        comparison = {
            'your_score': your_score,
            'baseline_score': baseline_score,
            'difference': your_score - baseline_score,
            'relative_improvement': (your_score - baseline_score) / baseline_score * 100
        }
        
        print(f"\nComparison with {baseline}:")
        print(f"  Your model:     {your_score:.3f}")
        print(f"  {baseline}: {baseline_score:.3f}")
        print(f"  Difference:     {comparison['difference']:+.3f}")
        print(f"  Relative:       {comparison['relative_improvement']:+.1f}%")
        
        return comparison


def run_mteb_evaluation(
    encoder,
    quick_eval: bool = True,
    categories: Optional[List[str]] = None,
    output_dir: str = 'results/mteb/'
) -> Dict[str, Any]:
    """
    Convenience function to run MTEB evaluation.
    
    Args:
        encoder: Text encoder to evaluate
        quick_eval: Run quick subset (True) or full evaluation (False)
        categories: Specific categories to evaluate
        output_dir: Output directory for results
        
    Returns:
        Evaluation results
    """
    print("\n" + "="*80)
    print("MTEB EVALUATION")
    print("="*80)
    
    evaluator = MTEBEvaluator(encoder)
    
    if categories:
        # Run specific categories
        all_results = {}
        for category in categories:
            print(f"\nEvaluating {category} tasks...")
            results = evaluator.run_category(category, output_dir)
            all_results[category] = results
        return all_results
    
    elif quick_eval:
        # Run quick evaluation
        print("\nRunning quick evaluation (5 diverse tasks)...")
        return evaluator.run(output_folder=output_dir)
    
    else:
        # Run full evaluation (this takes a long time!)
        print("\nRunning FULL evaluation (this will take several hours)...")
        all_tasks = []
        for tasks in MTEBEvaluator.TASK_CATEGORIES.values():
            all_tasks.extend(tasks)
        return evaluator.run(tasks=all_tasks, output_folder=output_dir)


if __name__ == "__main__":
    """Test MTEB evaluation with a sample encoder."""
    
    print("Testing MTEB Evaluation Adapter...")
    
    # Import encoder
    sys.path.insert(0, 'src')
    from Encoders import Text_to_Vec
    
    # Load encoder
    print("\nLoading Text_to_Vec encoder...")
    encoder = Text_to_Vec()
    
    # Run quick evaluation
    print("\nRunning quick MTEB evaluation...")
    results = run_mteb_evaluation(
        encoder,
        quick_eval=True,
        output_dir='results/mteb_test/'
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    if 'task_results' in results:
        print(f"\nOverall Score: {results['overall_score']:.3f}")
        print("\nTask Results:")
        for task, task_results in results['task_results'].items():
            print(f"  {task:30s} {task_results['metric']:10s}: {task_results['score']:.3f}")
        
        # Create submission file
        evaluator = MTEBEvaluator(encoder)
        evaluator.create_submission(
            results,
            'results/mteb_test/submission.json',
            model_name='Text_to_Vec_BART'
        )
        
        # Compare with baseline
        evaluator.compare_with_baseline(results, 'all-MiniLM-L6-v2')
    
    print("\nMTEB evaluation test complete!")