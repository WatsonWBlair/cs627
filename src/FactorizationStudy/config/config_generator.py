"""
Configuration Generator for Factorization Study.

Generates all experiment combinations based on study configuration parameters:
- 4 semantic dims x 3 hidden sizes x 4 layer counts = 48 base configs
- + random init variants
- + cross-transfer variants
"""

from typing import List, Optional, Iterator
from itertools import product

from .study_config import (
    StudyConfig,
    ExperimentConfig,
    AdapterConfig,
    InitStrategy,
    StudyPreset,
)


class ConfigGenerator:
    """Generates experiment configurations for a factorization study."""

    def __init__(self, study_config: StudyConfig):
        """
        Initialize the config generator.

        Args:
            study_config: Study configuration with sweep parameters
        """
        self.study_config = study_config

    def generate(self) -> List[ExperimentConfig]:
        """
        Generate all experiment configurations based on study parameters.

        Returns:
            List of ExperimentConfig objects
        """
        experiments = []
        exp_counter = 0

        # Generate base experiments: dims x hidden_sizes x hidden_layers x init_strategies
        for semantic_dim, hidden_size, hidden_layers, init_strategy in product(
            self.study_config.semantic_dims,
            self.study_config.hidden_sizes,
            self.study_config.hidden_layers,
            self.study_config.init_strategies,
        ):
            adapter_config = AdapterConfig(
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
            )

            experiment_id = self._generate_experiment_id(
                exp_counter, semantic_dim, adapter_config, init_strategy
            )

            experiment = ExperimentConfig(
                experiment_id=experiment_id,
                semantic_dim=semantic_dim,
                encoder_adapter=adapter_config,
                decoder_adapter=None,  # Use same as encoder
                init_strategy=init_strategy,
            )

            experiments.append(experiment)
            exp_counter += 1

        # Add cross-transfer experiments if requested
        if InitStrategy.CROSS_TRANSFER in self.study_config.init_strategies:
            cross_transfer_experiments = self._generate_cross_transfer_experiments(
                exp_counter
            )
            experiments.extend(cross_transfer_experiments)

        # Store in study config
        self.study_config.experiments = experiments

        return experiments

    def _generate_experiment_id(
        self,
        counter: int,
        semantic_dim: int,
        adapter_config: AdapterConfig,
        init_strategy: InitStrategy,
    ) -> str:
        """Generate a unique experiment ID."""
        init_suffix = {
            InitStrategy.PRETRAINED: "pre",
            InitStrategy.RANDOM: "rnd",
            InitStrategy.CROSS_TRANSFER: "xfr",
        }.get(init_strategy, "unk")

        return (
            f"exp_{counter:04d}_"
            f"dim{semantic_dim}_"
            f"h{adapter_config.hidden_size}_"
            f"l{adapter_config.hidden_layers}_"
            f"{init_suffix}"
        )

    def _generate_cross_transfer_experiments(
        self, start_counter: int
    ) -> List[ExperimentConfig]:
        """
        Generate cross-transfer experiments.

        Cross-transfer means initializing one modality's encoder adapter
        with weights from another modality's trained adapter.

        Transfer pairs:
        - text -> audio (text semantic knowledge to audio)
        - text -> image (text semantic knowledge to image)
        - audio -> image (audio features to image)
        """
        experiments = []
        counter = start_counter

        transfer_pairs = [
            ("text", "audio"),
            ("text", "image"),
            ("audio", "image"),
        ]

        # Only generate for default adapter config and each semantic dim
        default_adapter = AdapterConfig(hidden_size=200, hidden_layers=2)

        for semantic_dim in self.study_config.semantic_dims:
            for source, target in transfer_pairs:
                experiment_id = (
                    f"exp_{counter:04d}_"
                    f"dim{semantic_dim}_"
                    f"xfr_{source}_to_{target}"
                )

                experiment = ExperimentConfig(
                    experiment_id=experiment_id,
                    semantic_dim=semantic_dim,
                    encoder_adapter=default_adapter,
                    init_strategy=InitStrategy.CROSS_TRANSFER,
                    cross_transfer_source=source,
                    # Only train the target modality
                    train_text=(target == "text"),
                    train_audio=(target == "audio"),
                    train_image=(target == "image"),
                )

                experiments.append(experiment)
                counter += 1

        return experiments

    def get_experiment_count(self) -> int:
        """Get the total number of experiments that will be generated."""
        if self.study_config.experiments:
            return len(self.study_config.experiments)

        # Calculate without generating
        base_count = (
            len(self.study_config.semantic_dims)
            * len(self.study_config.hidden_sizes)
            * len(self.study_config.hidden_layers)
            * len(self.study_config.init_strategies)
        )

        # Cross-transfer adds experiments
        if InitStrategy.CROSS_TRANSFER in self.study_config.init_strategies:
            base_count += len(self.study_config.semantic_dims) * 3  # 3 transfer pairs

        return base_count

    def iterate_experiments(self) -> Iterator[ExperimentConfig]:
        """Iterate over experiments, generating if needed."""
        if not self.study_config.experiments:
            self.generate()

        yield from self.study_config.experiments

    def get_experiments_by_dimension(self, dim: int) -> List[ExperimentConfig]:
        """Get all experiments with a specific semantic dimension."""
        if not self.study_config.experiments:
            self.generate()

        return [e for e in self.study_config.experiments if e.semantic_dim == dim]

    def get_experiments_by_adapter(
        self, hidden_size: int, hidden_layers: int
    ) -> List[ExperimentConfig]:
        """Get all experiments with a specific adapter configuration."""
        if not self.study_config.experiments:
            self.generate()

        return [
            e for e in self.study_config.experiments
            if e.encoder_adapter.hidden_size == hidden_size
            and e.encoder_adapter.hidden_layers == hidden_layers
        ]

    def get_experiments_by_init_strategy(
        self, strategy: InitStrategy
    ) -> List[ExperimentConfig]:
        """Get all experiments with a specific initialization strategy."""
        if not self.study_config.experiments:
            self.generate()

        return [e for e in self.study_config.experiments if e.init_strategy == strategy]

    def summary(self) -> str:
        """Generate a summary of the experiment configurations."""
        if not self.study_config.experiments:
            self.generate()

        lines = [
            f"Factorization Study: {self.study_config.study_name}",
            f"Preset: {self.study_config.preset.value}",
            f"Total Experiments: {len(self.study_config.experiments)}",
            "",
            "Parameter Ranges:",
            f"  Semantic Dimensions: {self.study_config.semantic_dims}",
            f"  Hidden Sizes: {self.study_config.hidden_sizes}",
            f"  Hidden Layers: {self.study_config.hidden_layers}",
            f"  Init Strategies: {[s.value for s in self.study_config.init_strategies]}",
            "",
            "Experiments by Dimension:",
        ]

        for dim in self.study_config.semantic_dims:
            count = len(self.get_experiments_by_dimension(dim))
            lines.append(f"  {dim}D: {count} experiments")

        lines.extend([
            "",
            "Training Defaults:",
            f"  Epochs: {self.study_config.default_epochs}",
            f"  Batch Size: {self.study_config.default_batch_size}",
            f"  Learning Rate: {self.study_config.default_learning_rate}",
        ])

        return "\n".join(lines)


def generate_quick_configs(study_name: str = "quick_test") -> StudyConfig:
    """Generate a quick test configuration with minimal experiments."""
    from .study_config import create_study_config

    config = create_study_config(study_name, preset=StudyPreset.QUICK)
    generator = ConfigGenerator(config)
    generator.generate()

    return config


def generate_full_sweep_configs(study_name: str = "full_sweep") -> StudyConfig:
    """Generate a full sweep configuration with all experiments."""
    from .study_config import create_study_config

    config = create_study_config(study_name, preset=StudyPreset.FULL_SWEEP)
    generator = ConfigGenerator(config)
    generator.generate()

    return config
