"""
GLUE (General Language Understanding Evaluation) Benchmark Adapter

This module provides an adapter to evaluate trained encoders on GLUE benchmark tasks.
GLUE consists of 9 diverse natural language understanding tasks:

1. CoLA - Corpus of Linguistic Acceptability (single sentence, acceptability classification)
2. SST-2 - Stanford Sentiment Treebank (single sentence, sentiment classification)
3. MRPC - Microsoft Research Paraphrase Corpus (sentence pair, paraphrase detection)
4. QQP - Quora Question Pairs (sentence pair, duplicate detection)
5. STS-B - Semantic Textual Similarity Benchmark (sentence pair, similarity regression)
6. MNLI - Multi-Genre Natural Language Inference (sentence pair, entailment classification)
7. QNLI - Question Natural Language Inference (sentence pair, entailment classification)
8. RTE - Recognizing Textual Entailment (sentence pair, entailment classification)
9. WNLI - Winograd Natural Language Inference (sentence pair, coreference/entailment)

The adapter trains lightweight task heads on top of frozen encoder features.

Usage:
    from src.Evaluation.benchmarks.glue_adapter import GLUEEvaluator
    from src.Encoders import Text_to_Vec
    
    encoder = Text_to_Vec()
    evaluator = GLUEEvaluator(encoder)
    
    # Run on specific tasks
    results = evaluator.run(tasks=['sst2', 'mrpc', 'stsb'])
    
    # Run on all tasks
    results = evaluator.run_all()
    
    # Generate GLUE submission
    evaluator.create_submission(results, 'results/glue_submission.tsv')
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# HuggingFace imports
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace datasets not installed. Install with: pip install datasets transformers")


class GLUEDataset(Dataset):
    """PyTorch dataset for GLUE tasks."""
    
    def __init__(self, texts1: List[str], texts2: Optional[List[str]], labels: List, task: str):
        """
        Initialize GLUE dataset.
        
        Args:
            texts1: First text inputs (or only text for single-sentence tasks)
            texts2: Second text inputs (for sentence-pair tasks)
            labels: Task labels
            task: GLUE task name
        """
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.task = task
        self.is_pair_task = texts2 is not None
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.is_pair_task:
            return self.texts1[idx], self.texts2[idx], self.labels[idx]
        else:
            return self.texts1[idx], self.labels[idx]


class TaskHead(nn.Module):
    """Lightweight task-specific head for GLUE tasks."""
    
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        is_regression: bool = False
    ):
        """
        Initialize task head.
        
        Args:
            input_dim: Input dimension (encoder output size)
            num_labels: Number of output labels
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            is_regression: Whether this is a regression task
        """
        super().__init__()
        self.is_regression = is_regression
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels if not is_regression else 1)
        )
        
    def forward(self, features):
        """Forward pass through task head."""
        output = self.classifier(features)
        if self.is_regression:
            output = output.squeeze(-1)
        return output


class GLUEEvaluator:
    """
    Evaluator for GLUE benchmark tasks.
    
    Trains task-specific heads on frozen encoder representations
    and evaluates on standard GLUE metrics.
    """
    
    # Task configurations
    TASK_CONFIGS = {
        'cola': {
            'num_labels': 2,
            'metric': 'matthews_correlation',
            'is_regression': False,
            'single_sentence': True
        },
        'sst2': {
            'num_labels': 2,
            'metric': 'accuracy',
            'is_regression': False,
            'single_sentence': True
        },
        'mrpc': {
            'num_labels': 2,
            'metric': 'f1_accuracy',
            'is_regression': False,
            'single_sentence': False
        },
        'qqp': {
            'num_labels': 2,
            'metric': 'f1_accuracy',
            'is_regression': False,
            'single_sentence': False
        },
        'stsb': {
            'num_labels': 1,
            'metric': 'pearson_spearman',
            'is_regression': True,
            'single_sentence': False
        },
        'mnli': {
            'num_labels': 3,
            'metric': 'accuracy',
            'is_regression': False,
            'single_sentence': False
        },
        'qnli': {
            'num_labels': 2,
            'metric': 'accuracy',
            'is_regression': False,
            'single_sentence': False
        },
        'rte': {
            'num_labels': 2,
            'metric': 'accuracy',
            'is_regression': False,
            'single_sentence': False
        },
        'wnli': {
            'num_labels': 2,
            'metric': 'accuracy',
            'is_regression': False,
            'single_sentence': False
        }
    }
    
    def __init__(
        self,
        encoder,
        encoder_dim: int = 1024,
        hidden_dim: int = 256,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: str = None,
        freeze_encoder: bool = True
    ):
        """
        Initialize GLUE evaluator.
        
        Args:
            encoder: Text encoder to evaluate
            encoder_dim: Encoder output dimension
            hidden_dim: Task head hidden dimension
            batch_size: Training batch size
            learning_rate: Learning rate for task heads
            num_epochs: Number of training epochs per task
            device: Device for computation
            freeze_encoder: Whether to freeze encoder during training
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace not installed. Run: pip install datasets transformers")
        
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.freeze_encoder = freeze_encoder
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Move encoder to device
        self.encoder.to(device)
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def load_task_data(self, task: str) -> Tuple[GLUEDataset, GLUEDataset, Optional[GLUEDataset]]:
        """
        Load data for a GLUE task.
        
        Args:
            task: Task name
            
        Returns:
            Train, validation, and test datasets
        """
        print(f"Loading {task.upper()} data...")
        
        # Load from HuggingFace
        dataset = load_dataset('glue', task)
        
        config = self.TASK_CONFIGS[task]
        is_single = config['single_sentence']
        
        # Extract data
        def extract_data(split_data):
            if is_single:
                if task == 'cola':
                    texts1 = split_data['sentence']
                elif task == 'sst2':
                    texts1 = split_data['sentence']
                else:
                    texts1 = split_data.get('sentence', split_data.get('text', []))
                texts2 = None
            else:
                if task in ['mrpc', 'stsb']:
                    texts1 = split_data['sentence1']
                    texts2 = split_data['sentence2']
                elif task == 'qqp':
                    texts1 = split_data['question1']
                    texts2 = split_data['question2']
                elif task in ['mnli', 'rte', 'wnli']:
                    texts1 = split_data['premise']
                    texts2 = split_data['hypothesis']
                elif task == 'qnli':
                    texts1 = split_data['question']
                    texts2 = split_data['sentence']
                else:
                    raise ValueError(f"Unknown task format: {task}")
            
            labels = split_data['label']
            return texts1, texts2, labels
        
        # Create datasets
        train_texts1, train_texts2, train_labels = extract_data(dataset['train'])
        train_dataset = GLUEDataset(train_texts1, train_texts2, train_labels, task)
        
        # Validation set
        val_split = 'validation_matched' if task == 'mnli' else 'validation'
        if val_split in dataset:
            val_texts1, val_texts2, val_labels = extract_data(dataset[val_split])
            val_dataset = GLUEDataset(val_texts1, val_texts2, val_labels, task)
        else:
            val_dataset = None
        
        # Test set (usually unlabeled)
        test_split = 'test_matched' if task == 'mnli' else 'test'
        if test_split in dataset and 'label' in dataset[test_split].features:
            test_texts1, test_texts2, test_labels = extract_data(dataset[test_split])
            test_dataset = GLUEDataset(test_texts1, test_texts2, test_labels, task)
        else:
            test_dataset = None
        
        print(f"  Train: {len(train_dataset)} samples")
        if val_dataset:
            print(f"  Val: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def encode_texts(self, texts: Union[List[str], Tuple[List[str], List[str]]]) -> torch.Tensor:
        """
        Encode texts using the encoder.
        
        Args:
            texts: Single list of texts or tuple of (texts1, texts2) for pair tasks
            
        Returns:
            Encoded representations
        """
        with torch.no_grad():
            if isinstance(texts, tuple):
                # Sentence pair task - encode separately and concatenate
                texts1, texts2 = texts
                enc1 = self.encoder(texts1, device=self.device)
                enc2 = self.encoder(texts2, device=self.device)
                # Concatenate, difference, and element-wise product
                features = torch.cat([enc1, enc2, enc1 - enc2, enc1 * enc2], dim=1)
            else:
                # Single sentence task
                features = self.encoder(texts, device=self.device)
        
        return features
    
    def train_task_head(
        self,
        task: str,
        train_dataset: GLUEDataset,
        val_dataset: Optional[GLUEDataset]
    ) -> Tuple[TaskHead, Dict[str, float]]:
        """
        Train a task-specific head.
        
        Args:
            task: Task name
            train_dataset: Training data
            val_dataset: Validation data
            
        Returns:
            Trained task head and best validation metrics
        """
        config = self.TASK_CONFIGS[task]
        
        # Adjust input dimension for pair tasks (4x due to concatenation)
        input_dim = self.encoder_dim * 4 if not config['single_sentence'] else self.encoder_dim
        
        # Create task head
        task_head = TaskHead(
            input_dim=input_dim,
            num_labels=config['num_labels'],
            hidden_dim=self.hidden_dim,
            is_regression=config['is_regression']
        ).to(self.device)
        
        # Create optimizer
        optimizer = optim.AdamW(task_head.parameters(), lr=self.learning_rate)
        
        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Loss function
        if config['is_regression']:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_score = float('-inf')
        best_metrics = {}
        
        print(f"\nTraining {task.upper()} task head...")
        for epoch in range(self.num_epochs):
            task_head.train()
            total_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in progress_bar:
                if config['single_sentence']:
                    texts, labels = batch
                    features = self.encode_texts(texts)
                else:
                    texts1, texts2, labels = batch
                    features = self.encode_texts((texts1, texts2))
                
                # Move to device
                labels = torch.tensor(labels).to(self.device)
                if config['is_regression']:
                    labels = labels.float()
                
                # Forward pass
                outputs = task_head(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Validation
            if val_dataset:
                val_metrics = self.evaluate_task(task, task_head, val_dataset)
                val_score = self._get_primary_metric(val_metrics, config['metric'])
                
                print(f"  Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Val={val_score:.4f}")
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_metrics = val_metrics
        
        return task_head, best_metrics
    
    def evaluate_task(
        self,
        task: str,
        task_head: TaskHead,
        dataset: GLUEDataset
    ) -> Dict[str, float]:
        """
        Evaluate a task head on a dataset.
        
        Args:
            task: Task name
            task_head: Trained task head
            dataset: Evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        config = self.TASK_CONFIGS[task]
        task_head.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if config['single_sentence']:
                    texts, labels = batch
                    features = self.encode_texts(texts)
                else:
                    texts1, texts2, labels = batch
                    features = self.encode_texts((texts1, texts2))
                
                outputs = task_head(features)
                
                if config['is_regression']:
                    preds = outputs.cpu().numpy()
                else:
                    preds = outputs.argmax(dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        return self._calculate_metrics(task, all_preds, all_labels)
    
    def _calculate_metrics(self, task: str, preds: List, labels: List) -> Dict[str, float]:
        """Calculate task-specific metrics."""
        config = self.TASK_CONFIGS[task]
        metrics = {}
        
        if config['metric'] == 'accuracy':
            metrics['accuracy'] = accuracy_score(labels, preds)
            
        elif config['metric'] == 'f1_accuracy':
            metrics['f1'] = f1_score(labels, preds)
            metrics['accuracy'] = accuracy_score(labels, preds)
            
        elif config['metric'] == 'matthews_correlation':
            metrics['matthews_corr'] = matthews_corrcoef(labels, preds)
            
        elif config['metric'] == 'pearson_spearman':
            pearson_corr, _ = pearsonr(preds, labels)
            spearman_corr, _ = spearmanr(preds, labels)
            metrics['pearson'] = pearson_corr
            metrics['spearman'] = spearman_corr
        
        return metrics
    
    def _get_primary_metric(self, metrics: Dict[str, float], metric_name: str) -> float:
        """Get primary metric value for a task."""
        if metric_name == 'accuracy':
            return metrics.get('accuracy', 0.0)
        elif metric_name == 'f1_accuracy':
            return (metrics.get('f1', 0.0) + metrics.get('accuracy', 0.0)) / 2
        elif metric_name == 'matthews_correlation':
            return metrics.get('matthews_corr', 0.0)
        elif metric_name == 'pearson_spearman':
            return (metrics.get('pearson', 0.0) + metrics.get('spearman', 0.0)) / 2
        else:
            return 0.0
    
    def _collate_fn(self, batch):
        """Custom collate function for dataloaders."""
        if len(batch[0]) == 2:  # Single sentence
            texts = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            return texts, labels
        else:  # Sentence pair
            texts1 = [item[0] for item in batch]
            texts2 = [item[1] for item in batch]
            labels = [item[2] for item in batch]
            return texts1, texts2, labels
    
    def run(
        self,
        tasks: Optional[List[str]] = None,
        save_heads: bool = True,
        output_dir: str = 'results/glue/'
    ) -> Dict[str, Dict[str, float]]:
        """
        Run evaluation on specified GLUE tasks.
        
        Args:
            tasks: List of task names (None = all tasks)
            save_heads: Whether to save trained task heads
            output_dir: Directory to save results
            
        Returns:
            Results dictionary by task
        """
        if tasks is None:
            tasks = list(self.TASK_CONFIGS.keys())
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        task_heads = {}
        
        print("\n" + "="*80)
        print("GLUE EVALUATION")
        print("="*80)
        print(f"Tasks: {tasks}")
        
        for task in tasks:
            print(f"\n{'='*40}")
            print(f"Task: {task.upper()}")
            print('='*40)
            
            # Load data
            train_dataset, val_dataset, test_dataset = self.load_task_data(task)
            
            # Train task head
            task_head, val_metrics = self.train_task_head(task, train_dataset, val_dataset)
            task_heads[task] = task_head
            
            # Store results
            results[task] = val_metrics
            
            # Save task head
            if save_heads:
                head_path = output_path / f'{task}_head.pt'
                torch.save(task_head.state_dict(), head_path)
                print(f"Saved task head to {head_path}")
        
        # Calculate average score
        all_scores = []
        for task, metrics in results.items():
            config = self.TASK_CONFIGS[task]
            score = self._get_primary_metric(metrics, config['metric'])
            all_scores.append(score)
        
        results['average'] = {'score': np.mean(all_scores)}
        
        # Save results
        results_file = output_path / 'glue_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        for task, metrics in results.items():
            if task == 'average':
                print(f"\nAverage Score: {metrics['score']:.3f}")
            else:
                print(f"{task.upper():6s}: {metrics}")
        
        return results
    
    def run_all(self, save_heads: bool = True, output_dir: str = 'results/glue/') -> Dict:
        """Run evaluation on all GLUE tasks."""
        return self.run(tasks=None, save_heads=save_heads, output_dir=output_dir)
    
    def create_submission(
        self,
        results: Dict[str, Dict[str, float]],
        output_file: str = 'glue_submission.json'
    ):
        """
        Create GLUE submission file.
        
        Args:
            results: Results from run()
            output_file: Path to save submission
        """
        submission = {
            'results': {},
            'metadata': {
                'model': 'SemanticVectorSpace',
                'encoder_dim': self.encoder_dim,
                'tasks_evaluated': list(results.keys())
            }
        }
        
        # Format results
        for task, metrics in results.items():
            if task != 'average':
                config = self.TASK_CONFIGS[task]
                primary_metric = self._get_primary_metric(metrics, config['metric'])
                submission['results'][task] = {
                    'primary_metric': primary_metric,
                    'all_metrics': metrics
                }
        
        # Add average
        if 'average' in results:
            submission['average_score'] = results['average']['score']
        
        # Save submission
        with open(output_file, 'w') as f:
            json.dump(submission, f, indent=2)
        
        print(f"\nSubmission file created: {output_file}")
        print(f"Average score: {submission.get('average_score', 0.0):.3f}")


def run_glue_evaluation(
    encoder,
    tasks: Optional[List[str]] = None,
    output_dir: str = 'results/glue/'
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to run GLUE evaluation.
    
    Args:
        encoder: Text encoder to evaluate
        tasks: List of tasks to evaluate (None = all)
        output_dir: Output directory
        
    Returns:
        Evaluation results
    """
    evaluator = GLUEEvaluator(encoder)
    return evaluator.run(tasks=tasks, output_dir=output_dir)


if __name__ == "__main__":
    """Test GLUE evaluation with a sample encoder."""
    
    print("Testing GLUE Evaluation Adapter...")
    
    # Import encoder
    sys.path.insert(0, 'src')
    from Encoders import Text_to_Vec
    
    # Load encoder
    print("\nLoading Text_to_Vec encoder...")
    encoder = Text_to_Vec()
    
    # Run evaluation on subset of tasks
    print("\nRunning GLUE evaluation on SST-2 and MRPC...")
    results = run_glue_evaluation(
        encoder,
        tasks=['sst2', 'mrpc'],
        output_dir='results/glue_test/'
    )
    
    print("\nGLUE evaluation test complete!")