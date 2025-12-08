"""
Adapter Configuration Generator for Ablation Studies

Generates different adapter configurations to systematically test
the impact of architectural choices on performance.
"""

import itertools
from typing import List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class AdapterConfig:
    """Configuration for an adapter architecture."""
    name: str
    hidden_size: int
    num_layers: int
    activation: str
    dropout: float
    input_dim: int = 1024
    output_dim: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }
    
    def get_identifier(self) -> str:
        """Get unique identifier for this configuration."""
        return f"h{self.hidden_size}_l{self.num_layers}_{self.activation}_d{int(self.dropout*10)}"


class ConfigGenerator:
    """Generates adapter configurations for ablation studies."""
    
    # Default parameter ranges
    DEFAULT_PARAMS = {
        'hidden_sizes': [128, 256, 512, 1024],
        'num_layers': [1, 2, 3, 4],
        'activations': ['relu', 'gelu', 'silu'],
        'dropout_rates': [0.0, 0.1, 0.2, 0.3]
    }
    
    # Recommended configurations based on preliminary testing
    RECOMMENDED_CONFIGS = [
        {'hidden_size': 512, 'num_layers': 2, 'activation': 'relu', 'dropout': 0.1},
        {'hidden_size': 256, 'num_layers': 3, 'activation': 'gelu', 'dropout': 0.1},
        {'hidden_size': 1024, 'num_layers': 1, 'activation': 'relu', 'dropout': 0.0},
    ]
    
    def __init__(self, param_ranges: Dict[str, List] = None):
        """
        Initialize configuration generator.
        
        Args:
            param_ranges: Custom parameter ranges (uses defaults if None)
        """
        self.param_ranges = param_ranges or self.DEFAULT_PARAMS
    
    def generate_grid_search(self) -> List[AdapterConfig]:
        """
        Generate all combinations for grid search.
        
        Returns:
            List of adapter configurations
        """
        configs = []
        
        # Generate all combinations
        combinations = itertools.product(
            self.param_ranges['hidden_sizes'],
            self.param_ranges['num_layers'],
            self.param_ranges['activations'],
            self.param_ranges['dropout_rates']
        )
        
        for hidden_size, num_layers, activation, dropout in combinations:
            config = AdapterConfig(
                name=f"adapter_{len(configs)}",
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=activation,
                dropout=dropout
            )
            configs.append(config)
        
        return configs
    
    def generate_random_search(self, n_configs: int = 20) -> List[AdapterConfig]:
        """
        Generate random configurations for random search.
        
        Args:
            n_configs: Number of configurations to generate
        
        Returns:
            List of adapter configurations
        """
        import random
        
        configs = []
        
        for i in range(n_configs):
            config = AdapterConfig(
                name=f"adapter_{i}",
                hidden_size=random.choice(self.param_ranges['hidden_sizes']),
                num_layers=random.choice(self.param_ranges['num_layers']),
                activation=random.choice(self.param_ranges['activations']),
                dropout=random.choice(self.param_ranges['dropout_rates'])
            )
            configs.append(config)
        
        return configs
    
    def generate_progressive_depth(self) -> List[AdapterConfig]:
        """
        Generate configurations with progressively increasing depth.
        
        Returns:
            List of adapter configurations
        """
        configs = []
        
        for num_layers in self.param_ranges['num_layers']:
            # Use optimal hidden size for each depth
            if num_layers == 1:
                hidden_size = 1024
            elif num_layers == 2:
                hidden_size = 512
            elif num_layers == 3:
                hidden_size = 256
            else:
                hidden_size = 128
            
            config = AdapterConfig(
                name=f"progressive_{num_layers}",
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation='relu',
                dropout=0.1
            )
            configs.append(config)
        
        return configs
    
    def generate_activation_study(self) -> List[AdapterConfig]:
        """
        Generate configurations to study activation functions.
        
        Returns:
            List of adapter configurations
        """
        configs = []
        
        # Fixed architecture, vary activation
        for activation in self.param_ranges['activations']:
            config = AdapterConfig(
                name=f"activation_{activation}",
                hidden_size=512,
                num_layers=2,
                activation=activation,
                dropout=0.1
            )
            configs.append(config)
        
        return configs
    
    def generate_dropout_study(self) -> List[AdapterConfig]:
        """
        Generate configurations to study dropout rates.
        
        Returns:
            List of adapter configurations
        """
        configs = []
        
        # Fixed architecture, vary dropout
        for dropout in self.param_ranges['dropout_rates']:
            config = AdapterConfig(
                name=f"dropout_{int(dropout*10)}",
                hidden_size=512,
                num_layers=2,
                activation='relu',
                dropout=dropout
            )
            configs.append(config)
        
        return configs
    
    def generate_recommended(self) -> List[AdapterConfig]:
        """
        Generate recommended configurations.
        
        Returns:
            List of recommended adapter configurations
        """
        configs = []
        
        for i, params in enumerate(self.RECOMMENDED_CONFIGS):
            config = AdapterConfig(
                name=f"recommended_{i}",
                **params
            )
            configs.append(config)
        
        return configs
    
    def save_configs(self, configs: List[AdapterConfig], output_path: str):
        """
        Save configurations to JSON file.
        
        Args:
            configs: List of adapter configurations
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dicts = [c.to_dict() for c in configs]
        
        with open(output_path, 'w') as f:
            json.dump({
                'num_configs': len(configs),
                'configs': config_dicts
            }, f, indent=2)
        
        print(f"Saved {len(configs)} configurations to {output_path}")
    
    @staticmethod
    def load_configs(config_path: str) -> List[AdapterConfig]:
        """
        Load configurations from JSON file.
        
        Args:
            config_path: Path to JSON file
        
        Returns:
            List of adapter configurations
        """
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        configs = []
        for config_dict in data['configs']:
            configs.append(AdapterConfig(**config_dict))
        
        return configs


def estimate_training_time(configs: List[AdapterConfig], epochs_per_config: int = 50) -> Dict:
    """
    Estimate training time for ablation study.
    
    Args:
        configs: List of adapter configurations
        epochs_per_config: Number of epochs per configuration
    
    Returns:
        Time estimates dictionary
    """
    # Rough estimates based on adapter complexity
    base_time_per_epoch = 0.5  # minutes for simple adapter
    
    total_time = 0
    for config in configs:
        # Time scales with model size
        complexity_factor = (
            (config.hidden_size / 256) * 
            (config.num_layers / 2) *
            (1 + config.dropout * 0.1)  # Dropout adds slight overhead
        )
        time_per_epoch = base_time_per_epoch * complexity_factor
        total_time += time_per_epoch * epochs_per_config
    
    return {
        'total_minutes': total_time,
        'total_hours': total_time / 60,
        'configs': len(configs),
        'epochs_per_config': epochs_per_config,
        'average_minutes_per_config': total_time / len(configs)
    }


def main():
    """Generate and save different configuration sets."""
    generator = ConfigGenerator()
    
    # Generate different configuration sets
    config_sets = {
        'grid_search': generator.generate_grid_search(),
        'random_search': generator.generate_random_search(30),
        'progressive_depth': generator.generate_progressive_depth(),
        'activation_study': generator.generate_activation_study(),
        'dropout_study': generator.generate_dropout_study(),
        'recommended': generator.generate_recommended()
    }
    
    # Save each configuration set
    output_dir = Path('configs/ablation')
    
    for name, configs in config_sets.items():
        output_path = output_dir / f"{name}_configs.json"
        generator.save_configs(configs, output_path)
        
        # Estimate training time
        estimates = estimate_training_time(configs)
        print(f"\n{name.upper()} Study:")
        print(f"  Configurations: {estimates['configs']}")
        print(f"  Estimated time: {estimates['total_hours']:.1f} hours")
        print(f"  Per config: {estimates['average_minutes_per_config']:.1f} minutes")
    
    print(f"\nAll configurations saved to {output_dir}/")


if __name__ == "__main__":
    main()