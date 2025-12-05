"""
Configuration dataclasses for Factorization Study.

Defines the configuration structures for:
- Adapter configurations (hidden size, layers)
- Individual experiment configurations
- Overall study configurations
- Study presets (quick, full_sweep, custom)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
from pathlib import Path


class InitStrategy(Enum):
    """Weight initialization strategies for encoders."""
    RANDOM = "random"           # Random initialization
    PRETRAINED = "pretrained"   # Use pretrained adapter weights
    CROSS_TRANSFER = "cross_transfer"  # Transfer from another modality


class StudyPreset(Enum):
    """Predefined study configurations."""
    QUICK = "quick"             # 4 experiments for testing
    STANDARD = "standard"       # 16 experiments (balanced)
    FULL_SWEEP = "full_sweep"   # ~60 experiments (comprehensive)
    CUSTOM = "custom"           # User-defined configuration


@dataclass
class AdapterConfig:
    """Configuration for an adapter module (MLP bridge)."""
    hidden_size: int = 200
    hidden_layers: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "hidden_layers": self.hidden_layers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterConfig":
        return cls(
            hidden_size=data.get("hidden_size", 200),
            hidden_layers=data.get("hidden_layers", 2),
        )

    def __str__(self) -> str:
        return f"h{self.hidden_size}_l{self.hidden_layers}"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment in the study."""
    experiment_id: str
    semantic_dim: int  # 256, 512, 1024, 2048
    encoder_adapter: AdapterConfig
    decoder_adapter: Optional[AdapterConfig] = None  # If None, uses encoder_adapter
    init_strategy: InitStrategy = InitStrategy.PRETRAINED
    cross_transfer_source: Optional[str] = None  # Modality to transfer from

    # Training hyperparameters (can override study defaults)
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None

    # Modalities to train
    train_text: bool = True
    train_audio: bool = True
    train_image: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "semantic_dim": self.semantic_dim,
            "encoder_adapter": self.encoder_adapter.to_dict(),
            "decoder_adapter": self.decoder_adapter.to_dict() if self.decoder_adapter else None,
            "init_strategy": self.init_strategy.value,
            "cross_transfer_source": self.cross_transfer_source,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "train_text": self.train_text,
            "train_audio": self.train_audio,
            "train_image": self.train_image,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_id=data["experiment_id"],
            semantic_dim=data["semantic_dim"],
            encoder_adapter=AdapterConfig.from_dict(data["encoder_adapter"]),
            decoder_adapter=AdapterConfig.from_dict(data["decoder_adapter"]) if data.get("decoder_adapter") else None,
            init_strategy=InitStrategy(data.get("init_strategy", "pretrained")),
            cross_transfer_source=data.get("cross_transfer_source"),
            epochs=data.get("epochs"),
            batch_size=data.get("batch_size"),
            learning_rate=data.get("learning_rate"),
            train_text=data.get("train_text", True),
            train_audio=data.get("train_audio", True),
            train_image=data.get("train_image", True),
        )

    def get_effective_decoder_adapter(self) -> AdapterConfig:
        """Return decoder adapter config, defaulting to encoder adapter if not specified."""
        return self.decoder_adapter if self.decoder_adapter else self.encoder_adapter


@dataclass
class StudyConfig:
    """Overall configuration for a factorization study."""
    study_name: str
    output_dir: str = "studies"
    preset: StudyPreset = StudyPreset.STANDARD

    # Default training hyperparameters
    default_epochs: int = 30
    default_batch_size: int = 32
    default_learning_rate: float = 0.0001

    # Sweep parameters (used by ConfigGenerator)
    semantic_dims: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    hidden_sizes: List[int] = field(default_factory=lambda: [100, 200, 400])
    hidden_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    init_strategies: List[InitStrategy] = field(default_factory=lambda: [InitStrategy.PRETRAINED])

    # Evaluation settings
    eval_interval: int = 5  # Evaluate every N epochs
    save_checkpoints: bool = True
    checkpoint_interval: int = 10

    # Resource limits
    max_gpu_memory_gb: Optional[float] = None  # Warn if exceeded

    # Generated experiments (populated by ConfigGenerator)
    experiments: List[ExperimentConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "output_dir": self.output_dir,
            "preset": self.preset.value,
            "default_epochs": self.default_epochs,
            "default_batch_size": self.default_batch_size,
            "default_learning_rate": self.default_learning_rate,
            "semantic_dims": self.semantic_dims,
            "hidden_sizes": self.hidden_sizes,
            "hidden_layers": self.hidden_layers,
            "init_strategies": [s.value for s in self.init_strategies],
            "eval_interval": self.eval_interval,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_interval": self.checkpoint_interval,
            "max_gpu_memory_gb": self.max_gpu_memory_gb,
            "experiments": [e.to_dict() for e in self.experiments],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudyConfig":
        config = cls(
            study_name=data["study_name"],
            output_dir=data.get("output_dir", "studies"),
            preset=StudyPreset(data.get("preset", "standard")),
            default_epochs=data.get("default_epochs", 30),
            default_batch_size=data.get("default_batch_size", 32),
            default_learning_rate=data.get("default_learning_rate", 0.0001),
            semantic_dims=data.get("semantic_dims", [256, 512, 1024, 2048]),
            hidden_sizes=data.get("hidden_sizes", [100, 200, 400]),
            hidden_layers=data.get("hidden_layers", [1, 2, 3, 4]),
            init_strategies=[InitStrategy(s) for s in data.get("init_strategies", ["pretrained"])],
            eval_interval=data.get("eval_interval", 5),
            save_checkpoints=data.get("save_checkpoints", True),
            checkpoint_interval=data.get("checkpoint_interval", 10),
            max_gpu_memory_gb=data.get("max_gpu_memory_gb"),
        )
        config.experiments = [ExperimentConfig.from_dict(e) for e in data.get("experiments", [])]
        return config

    def save(self, path: Optional[str] = None) -> str:
        """Save configuration to JSON file."""
        if path is None:
            study_dir = Path(self.output_dir) / self.study_name
            study_dir.mkdir(parents=True, exist_ok=True)
            path = str(study_dir / "study_config.json")

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path

    @classmethod
    def load(cls, path: str) -> "StudyConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_study_dir(self) -> Path:
        """Get the study output directory."""
        return Path(self.output_dir) / self.study_name

    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Get the directory for a specific experiment."""
        return self.get_study_dir() / "experiments" / experiment_id


# Preset configurations
PRESET_CONFIGS = {
    StudyPreset.QUICK: {
        "semantic_dims": [512, 1024],
        "hidden_sizes": [200],
        "hidden_layers": [2],
        "init_strategies": [InitStrategy.PRETRAINED],
        "default_epochs": 10,
    },
    StudyPreset.STANDARD: {
        "semantic_dims": [256, 512, 1024, 2048],
        "hidden_sizes": [200],
        "hidden_layers": [2, 3],
        "init_strategies": [InitStrategy.PRETRAINED],
        "default_epochs": 30,
    },
    StudyPreset.FULL_SWEEP: {
        "semantic_dims": [256, 512, 1024, 2048],
        "hidden_sizes": [100, 200, 400],
        "hidden_layers": [1, 2, 3, 4],
        "init_strategies": [InitStrategy.PRETRAINED, InitStrategy.RANDOM],
        "default_epochs": 30,
    },
}


def create_study_config(
    study_name: str,
    preset: StudyPreset = StudyPreset.STANDARD,
    **overrides
) -> StudyConfig:
    """
    Create a study configuration with preset defaults.

    Args:
        study_name: Name of the study
        preset: Preset configuration to use
        **overrides: Override any preset values

    Returns:
        StudyConfig with preset and overridden values
    """
    # Start with preset defaults
    preset_config = PRESET_CONFIGS.get(preset, {})

    # Build config
    config = StudyConfig(
        study_name=study_name,
        preset=preset,
        semantic_dims=overrides.get("semantic_dims", preset_config.get("semantic_dims", [1024])),
        hidden_sizes=overrides.get("hidden_sizes", preset_config.get("hidden_sizes", [200])),
        hidden_layers=overrides.get("hidden_layers", preset_config.get("hidden_layers", [2])),
        init_strategies=overrides.get("init_strategies", preset_config.get("init_strategies", [InitStrategy.PRETRAINED])),
        default_epochs=overrides.get("default_epochs", preset_config.get("default_epochs", 30)),
        default_batch_size=overrides.get("default_batch_size", 32),
        default_learning_rate=overrides.get("default_learning_rate", 0.0001),
        output_dir=overrides.get("output_dir", "studies"),
    )

    return config
