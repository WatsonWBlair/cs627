"""
Ablation Study Runner

Systematically trains and evaluates different adapter configurations
to identify optimal architectures for the semantic vector space.

Usage:
    # Run recommended configurations
    python src/Experiments/run_ablation.py --config recommended
    
    # Run full grid search
    python src/Experiments/run_ablation.py --config grid_search
    
    # Run custom config file
    python src/Experiments/run_ablation.py --config-file configs/ablation/custom.json
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.Adapter import Adapter
from src.Training.Data_Wrangling.token_dataset import create_token_dataloader
from src.Training.adapter_trainer import AdapterTrainer, ContrastiveLoss
from src.Experiments.adapter_configs import ConfigGenerator, AdapterConfig


class AblationRunner:
    """Runs ablation studies on adapter configurations."""
    
    def __init__(
        self,
        token_dir: str = 'data/pregenerated_tokens/mosi',
        results_dir: str = 'Results/ablation',
        device: str = None
    ):
        """
        Initialize ablation runner.
        
        Args:
            token_dir: Directory containing pre-generated tokens
            results_dir: Directory to save results
            device: Device for training (auto-detect if None)
        """
        self.token_dir = Path(token_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data loaders once
        print("Loading token datasets...")
        self.train_loader = create_token_dataloader(
            'train',
            token_dir=str(self.token_dir),
            batch_size=256,
            shuffle=True,
            token_mode='matched',
            cache_size=2000
        )
        
        self.val_loader = create_token_dataloader(
            'val',
            token_dir=str(self.token_dir),
            batch_size=256,
            shuffle=False,
            token_mode='matched',
            cache_size=1000
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
    
    def create_adapter_from_config(
        self,
        config: AdapterConfig,
        encoder_name: str = 'text_base'
    ) -> Adapter:
        """
        Create adapter module from configuration.
        
        Args:
            config: Adapter configuration
            encoder_name: Name of encoder for this adapter
        
        Returns:
            Adapter module
        """
        # Map activation string to function
        activation_map = {
            'relu': torch.nn.ReLU,
            'gelu': torch.nn.GELU,
            'silu': torch.nn.SiLU,
            'tanh': torch.nn.Tanh
        }
        
        # Create custom adapter with specified architecture
        class CustomAdapter(Adapter):
            def __init__(self):
                super().__init__(
                    prefix=f"{encoder_name}_{config.get_identifier()}",
                    input_length=config.input_dim,
                    output_length=config.output_dim,
                    hidden_size=config.hidden_size,
                    hidden_layers=config.num_layers,
                    weights_dir="OptimalWeights"
                )
                
                # Rebuild model with custom activation and dropout
                layers = []
                
                # Input layer
                layers.append(torch.nn.Linear(config.input_dim, config.hidden_size))
                layers.append(activation_map[config.activation]())
                if config.dropout > 0:
                    layers.append(torch.nn.Dropout(config.dropout))
                
                # Hidden layers
                for _ in range(config.num_layers):
                    layers.append(torch.nn.Linear(config.hidden_size, config.hidden_size))
                    layers.append(activation_map[config.activation]())
                    if config.dropout > 0:
                        layers.append(torch.nn.Dropout(config.dropout))
                
                # Output layer
                layers.append(torch.nn.Linear(config.hidden_size, config.output_dim))
                
                self.model = torch.nn.Sequential(*layers)
        
        return CustomAdapter()
    
    def train_single_config(
        self,
        config: AdapterConfig,
        epochs: int = 30,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Train a single adapter configuration.
        
        Args:
            config: Adapter configuration to train
            epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Results dictionary
        """
        print(f"\nTraining {config.name}: {config.get_identifier()}")
        
        # Create adapters for each encoder type
        adapters = {
            'text_adapter': self.create_adapter_from_config(config, 'text_base'),
            'audio_adapter': self.create_adapter_from_config(config, 'audio_waveform'),
            'tone_adapter': self.create_adapter_from_config(config, 'audio_tone'),
            'image_adapter': self.create_adapter_from_config(config, 'image_base')
        }
        
        # Create trainer
        trainer = AdapterTrainer(
            adapters=adapters,
            loss_fn=ContrastiveLoss(temperature=0.07),
            learning_rate=learning_rate,
            device=self.device,
            use_amp=True,
            gradient_accumulation=1
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_recall_at_1': [],
            'val_recall_at_5': []
        }
        
        best_val_loss = float('inf')
        best_metrics = {}
        
        # Train
        for epoch in range(1, epochs + 1):
            # Train epoch
            train_loss = trainer.train_epoch(self.train_loader, epoch)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = trainer.validate(self.val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_recall_at_1'].append(val_metrics.get('recall@1', 0))
            history['val_recall_at_5'].append(val_metrics.get('recall@5', 0))
            
            # Track best
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_metrics = val_metrics.copy()
            
            # Early stopping
            if epoch > 10 and history['val_loss'][-1] > history['val_loss'][-10]:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        # Compute final statistics
        results = {
            'config': config.to_dict(),
            'identifier': config.get_identifier(),
            'epochs_trained': len(history['train_loss']),
            'best_val_loss': best_val_loss,
            'best_metrics': best_metrics,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_recall_at_1': history['val_recall_at_1'][-1],
            'final_recall_at_5': history['val_recall_at_5'][-1],
            'history': history
        }
        
        # Calculate model size
        total_params = sum(p.numel() for adapter in adapters.values() 
                          for p in adapter.parameters())
        results['total_parameters'] = total_params
        results['model_size_mb'] = total_params * 4 / (1024 * 1024)  # Float32
        
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Final R@1: {results['final_recall_at_1']:.3f}")
        print(f"  Model size: {results['model_size_mb']:.2f} MB")
        
        return results
    
    def run_ablation(
        self,
        configs: List[AdapterConfig],
        epochs_per_config: int = 30,
        learning_rate: float = 0.001
    ) -> pd.DataFrame:
        """
        Run ablation study on multiple configurations.
        
        Args:
            configs: List of adapter configurations
            epochs_per_config: Epochs to train each configuration
            learning_rate: Learning rate for all configs
        
        Returns:
            DataFrame with results
        """
        print(f"\n{'='*80}")
        print(f"Running ablation study on {len(configs)} configurations")
        print(f"Epochs per config: {epochs_per_config}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*80}")
        
        all_results = []
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Configuration: {config.name}")
            
            try:
                results = self.train_single_config(
                    config,
                    epochs=epochs_per_config,
                    learning_rate=learning_rate
                )
                all_results.append(results)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    'config': config.to_dict(),
                    'identifier': config.get_identifier(),
                    'error': str(e)
                })
        
        # Convert to DataFrame for analysis
        df_data = []
        for result in all_results:
            if 'error' not in result:
                df_data.append({
                    'name': result['config']['name'],
                    'identifier': result['identifier'],
                    'hidden_size': result['config']['hidden_size'],
                    'num_layers': result['config']['num_layers'],
                    'activation': result['config']['activation'],
                    'dropout': result['config']['dropout'],
                    'best_val_loss': result['best_val_loss'],
                    'final_recall_at_1': result['final_recall_at_1'],
                    'final_recall_at_5': result['final_recall_at_5'],
                    'epochs_trained': result['epochs_trained'],
                    'total_parameters': result['total_parameters'],
                    'model_size_mb': result['model_size_mb']
                })
        
        df = pd.DataFrame(df_data)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results as JSON
        json_path = self.results_dir / f"ablation_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save summary as CSV
        csv_path = self.results_dir / f"ablation_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*80}")
        print("Ablation study complete!")
        print(f"Detailed results: {json_path}")
        print(f"Summary CSV: {csv_path}")
        
        # Print top configurations
        if len(df) > 0:
            print("\nTop 5 configurations by validation loss:")
            top_configs = df.nsmallest(5, 'best_val_loss')
            print(top_configs[['identifier', 'best_val_loss', 'final_recall_at_1', 'model_size_mb']])
        
        return df
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze ablation study results.
        
        Args:
            df: DataFrame with ablation results
        
        Returns:
            Analysis dictionary
        """
        analysis = {}
        
        # Best overall configuration
        best_idx = df['best_val_loss'].idxmin()
        analysis['best_config'] = df.loc[best_idx].to_dict()
        
        # Impact of each hyperparameter
        for param in ['hidden_size', 'num_layers', 'activation', 'dropout']:
            param_impact = df.groupby(param)['best_val_loss'].agg(['mean', 'std', 'min'])
            analysis[f'{param}_impact'] = param_impact.to_dict()
        
        # Efficiency analysis (performance vs size)
        df['efficiency'] = df['final_recall_at_1'] / df['model_size_mb']
        best_efficiency_idx = df['efficiency'].idxmax()
        analysis['most_efficient'] = df.loc[best_efficiency_idx].to_dict()
        
        # Correlation analysis
        numeric_cols = ['hidden_size', 'num_layers', 'dropout', 
                       'best_val_loss', 'final_recall_at_1', 'model_size_mb']
        correlations = df[numeric_cols].corr()['best_val_loss'].to_dict()
        analysis['correlations'] = correlations
        
        return analysis


def main():
    """Main function to run ablation studies."""
    parser = argparse.ArgumentParser(description='Run ablation study on adapter configurations')
    parser.add_argument('--config', type=str, default='recommended',
                       choices=['grid_search', 'random_search', 'progressive_depth',
                               'activation_study', 'dropout_study', 'recommended'],
                       help='Configuration set to use')
    parser.add_argument('--config-file', type=str,
                       help='Path to custom configuration JSON file')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Epochs per configuration')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--token-dir', type=str, default='data/pregenerated_tokens/mosi',
                       help='Directory with pre-generated tokens')
    
    args = parser.parse_args()
    
    # Load configurations
    if args.config_file:
        configs = ConfigGenerator.load_configs(args.config_file)
    else:
        generator = ConfigGenerator()
        
        if args.config == 'grid_search':
            configs = generator.generate_grid_search()
        elif args.config == 'random_search':
            configs = generator.generate_random_search(20)
        elif args.config == 'progressive_depth':
            configs = generator.generate_progressive_depth()
        elif args.config == 'activation_study':
            configs = generator.generate_activation_study()
        elif args.config == 'dropout_study':
            configs = generator.generate_dropout_study()
        else:  # recommended
            configs = generator.generate_recommended()
    
    print(f"Loaded {len(configs)} configurations for {args.config} study")
    
    # Initialize runner
    runner = AblationRunner(token_dir=args.token_dir)
    
    # Run ablation
    results_df = runner.run_ablation(
        configs,
        epochs_per_config=args.epochs,
        learning_rate=args.lr
    )
    
    # Analyze results
    if len(results_df) > 0:
        analysis = runner.analyze_results(results_df)
        
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nBest configuration:")
        best = analysis['best_config']
        print(f"  {best['identifier']}")
        print(f"  Val loss: {best['best_val_loss']:.4f}")
        print(f"  Recall@1: {best['final_recall_at_1']:.3f}")
        print(f"  Size: {best['model_size_mb']:.2f} MB")
        
        print(f"\nMost efficient (performance/size):")
        efficient = analysis['most_efficient']
        print(f"  {efficient['identifier']}")
        print(f"  Efficiency: {efficient['efficiency']:.3f}")
        
        print(f"\nParameter correlations with val loss:")
        for param, corr in analysis['correlations'].items():
            if param != 'best_val_loss':
                print(f"  {param}: {corr:.3f}")


if __name__ == "__main__":
    main()