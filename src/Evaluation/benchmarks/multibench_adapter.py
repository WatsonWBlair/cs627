"""
MultiBench Adapter for Downstream Task Evaluation

Adapts our trained encoders to the MultiBench benchmark suite for evaluating
performance on downstream multimodal tasks.

MultiBench provides standardized evaluation across 20+ multimodal algorithms
on tasks like sentiment analysis, emotion recognition, and more.

Our approach:
1. Freeze pretrained encoders (Text_to_Vec, Audio_to_Vec, Image_to_Vec)
2. Train small task-specific heads on frozen encoder outputs
3. Evaluate on MultiBench tasks to measure semantic vector space quality

Reference:
    Liang et al. "MultiBench: Multiscale Benchmarks for Multimodal
    Representation Learning" (NeurIPS 2021)
    https://github.com/pliang279/MultiBench
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class TaskHead(nn.Module):
    """
    Lightweight task-specific head for downstream tasks.

    Takes concatenated multimodal embeddings and predicts task output.
    Kept small (1-2 layers) to ensure evaluation measures encoder quality,
    not head capacity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        task_type: str = "classification",
        dropout: float = 0.1
    ):
        """
        Initialize task head.

        Args:
            input_dim: Combined dimension of all encoder outputs
            output_dim: Number of output classes (classification) or 1 (regression)
            hidden_dim: Hidden layer dimension
            task_type: "classification" or "regression"
            dropout: Dropout probability
        """
        super().__init__()

        self.task_type = task_type

        # Simple 2-layer MLP
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, input_dim) concatenated embeddings

        Returns:
            (batch_size, output_dim) predictions
        """
        logits = self.head(x)

        if self.task_type == "regression":
            return logits.squeeze(-1)  # (batch_size,)
        else:
            return logits  # (batch_size, num_classes)


class MultiBenchAdapter(nn.Module):
    """
    Adapter for using pretrained encoders on MultiBench tasks.

    Freezes encoder weights and only trains a small task-specific head.
    This ensures evaluation measures the quality of the learned semantic
    vector space, not the capacity of the downstream model.
    """

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        task_type: str = "classification",
        num_classes: int = 7,
        hidden_dim: int = 128,
        fusion_method: str = "concat",
        freeze_encoders: bool = True
    ):
        """
        Initialize MultiBench adapter.

        Args:
            encoders: Dict mapping modality name to encoder module
            task_type: "classification" or "regression"
            num_classes: Number of output classes (for classification)
            hidden_dim: Hidden dimension for task head
            fusion_method: How to combine modality embeddings ("concat", "mean", "attention")
            freeze_encoders: Whether to freeze encoder weights
        """
        super().__init__()

        self.encoders = nn.ModuleDict(encoders)
        self.fusion_method = fusion_method
        self.task_type = task_type

        # Freeze encoders
        if freeze_encoders:
            for encoder in self.encoders.values():
                for param in encoder.parameters():
                    param.requires_grad = False
            print(f"[MultiBenchAdapter] Froze {len(encoders)} encoders")

        # Determine input dimension for task head
        # Assume all encoders output 1024-dim vectors
        encoder_dim = 1024
        num_encoders = len(encoders)

        if fusion_method == "concat":
            input_dim = encoder_dim * num_encoders
        elif fusion_method == "mean":
            input_dim = encoder_dim
        elif fusion_method == "attention":
            input_dim = encoder_dim
            # Attention weights for each modality
            self.attention_weights = nn.Parameter(torch.ones(num_encoders) / num_encoders)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Create task head
        output_dim = num_classes if task_type == "classification" else 1
        self.task_head = TaskHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            task_type=task_type
        )

        print(f"[MultiBenchAdapter] Created task head:")
        print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Task type: {task_type}")
        print(f"  Fusion: {fusion_method}")

    def forward(self, batch: Dict[str, any]) -> torch.Tensor:
        """
        Forward pass through frozen encoders + task head.

        Args:
            batch: Dict with keys matching encoder names
                   e.g., {"text": [...], "audio": tensor, "video": tensor}

        Returns:
            Task predictions (logits for classification, values for regression)
        """
        # Encode with each modality encoder
        embeddings = []

        for encoder_name, encoder in self.encoders.items():
            # Determine which data key to use
            if 'text' in encoder_name:
                data = batch.get('text')
            elif 'audio' in encoder_name:
                data = batch.get('audio')
            elif 'image' in encoder_name or 'visual' in encoder_name:
                data = batch.get('video') or batch.get('image')
            else:
                data = batch.get(encoder_name)

            if data is None:
                raise ValueError(f"No data found for encoder '{encoder_name}'")

            # Encode (encoders already frozen, but use no_grad for safety)
            with torch.no_grad():
                emb = encoder(data)

            embeddings.append(emb)

        # Fuse embeddings
        if self.fusion_method == "concat":
            # Concatenate all modalities
            fused = torch.cat(embeddings, dim=1)

        elif self.fusion_method == "mean":
            # Simple average
            fused = torch.stack(embeddings, dim=0).mean(dim=0)

        elif self.fusion_method == "attention":
            # Weighted average with learned attention weights
            weights = F.softmax(self.attention_weights, dim=0)
            fused = sum(w * emb for w, emb in zip(weights, embeddings))

        # Task-specific prediction
        output = self.task_head(fused)

        return output

    def train_on_task(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 20,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train task head on downstream task (encoders frozen).

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for task head
            device: Device to train on
            save_path: Optional path to save best model

        Returns:
            Dict with training history (loss, accuracy/mae)
        """
        self.to(device)
        self.train()

        # Optimizer only for task head (encoders frozen)
        optimizer = torch.optim.Adam(self.task_head.parameters(), lr=learning_rate)

        # Loss function
        if self.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": []  # accuracy or MAE
        }

        best_val_metric = float('inf') if self.task_type == "regression" else 0.0

        print(f"\n[MultiBenchAdapter] Training task head for {num_epochs} epochs...")
        print("=" * 80)

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                # Forward pass
                outputs = self.forward(batch)
                labels = batch['label'].to(device)

                # Compute loss
                if self.task_type == "classification":
                    loss = criterion(outputs, labels.long())
                else:
                    loss = criterion(outputs, labels.float())

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation phase
            self.eval()
            val_loss = 0.0
            val_metric = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.forward(batch)
                    labels = batch['label'].to(device)

                    # Loss
                    if self.task_type == "classification":
                        loss = criterion(outputs, labels.long())
                        # Accuracy
                        preds = outputs.argmax(dim=1)
                        val_metric += (preds == labels).float().mean().item()
                    else:
                        loss = criterion(outputs, labels.float())
                        # MAE
                        val_metric += (outputs - labels).abs().mean().item()

                    val_loss += loss.item()
                    num_val_batches += 1

            val_loss /= num_val_batches
            val_metric /= num_val_batches

            # Save history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metric"].append(val_metric)

            # Print progress
            metric_name = "Acc" if self.task_type == "classification" else "MAE"
            print(f"Epoch {epoch+1:2d}/{num_epochs}  "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Val Loss: {val_loss:.4f}  "
                  f"Val {metric_name}: {val_metric:.4f}")

            # Save best model
            is_best = (val_metric > best_val_metric if self.task_type == "classification"
                      else val_metric < best_val_metric)

            if is_best:
                best_val_metric = val_metric
                if save_path:
                    save_dir = Path(save_path).parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.task_head.state_dict(), save_path)
                    print(f"  → Saved best model to {save_path}")

        print("=" * 80)
        print(f"Training complete! Best val {metric_name}: {best_val_metric:.4f}")

        return history


class MOSISentimentTask(MultiBenchAdapter):
    """
    Specific adapter for CMU-MOSI sentiment analysis task.

    7-class sentiment classification: {-3, -2, -1, 0, 1, 2, 3}
    Maps continuous sentiment scores to discrete classes.
    """

    def __init__(self, encoders: Dict[str, nn.Module], **kwargs):
        """
        Initialize MOSI sentiment task adapter.

        Args:
            encoders: Dict of pretrained encoders
            **kwargs: Additional arguments for MultiBenchAdapter
        """
        # MOSI has 7 sentiment classes
        super().__init__(
            encoders=encoders,
            task_type="classification",
            num_classes=7,
            **kwargs
        )

    @staticmethod
    def sentiment_to_class(score: float) -> int:
        """
        Convert continuous sentiment score to discrete class.

        MOSI scores range from -3 (highly negative) to +3 (highly positive).

        Args:
            score: Continuous sentiment score

        Returns:
            Class index (0-6)
        """
        # Bins: [-3, -2), [-2, -1), [-1, 0), [0, 1), [1, 2), [2, 3), [3, inf)
        if score < -2.0:
            return 0
        elif score < -1.0:
            return 1
        elif score < 0.0:
            return 2
        elif score < 1.0:
            return 3
        elif score < 2.0:
            return 4
        elif score < 3.0:
            return 5
        else:
            return 6


if __name__ == "__main__":
    """
    Example usage with trained encoders on MOSI sentiment task.
    """
    import sys
    sys.path.insert(0, 'src')

    from Encoders import Text_to_Vec, Audio_to_Vec, Image_to_Vec

    print("Loading pretrained encoders...")
    encoders = {
        "text_base": Text_to_Vec(),
        "audio_waveform": Audio_to_Vec(),
        "image_base": Image_to_Vec()
    }

    print("Creating MOSI sentiment task adapter...")
    adapter = MOSISentimentTask(
        encoders=encoders,
        fusion_method="concat",
        hidden_dim=128,
        freeze_encoders=True
    )

    print(f"\nAdapter architecture:")
    print(f"  Encoder params (frozen): {sum(p.numel() for p in adapter.encoders.parameters()):,}")
    print(f"  Task head params (trainable): {sum(p.numel() for p in adapter.task_head.parameters()):,}")

    print("\nMultiBench adapter ready for training!")
    print("Next step: Load MOSI dataset and call adapter.train_on_task()")
