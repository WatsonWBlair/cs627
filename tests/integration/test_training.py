"""
Quick test of encoder training script with reduced epochs.
"""

import sys
import os

# Add src to path first
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now modify the training script to use fewer epochs for testing
from Training import train_encoders as training_module

# Store original values
ORIGINAL_EPOCHS = training_module.EPOCHS
ORIGINAL_BATCH_SIZE = training_module.BATCH_SIZE

# Set test values
training_module.EPOCHS = 2  # Just 2 epochs for testing
training_module.BATCH_SIZE = 4  # Smaller batch size

print("=" * 80)
print("TESTING MODE - Running with reduced parameters:")
print(f"  Epochs: {training_module.EPOCHS} (original: {ORIGINAL_EPOCHS})")
print(f"  Batch Size: {training_module.BATCH_SIZE} (original: {ORIGINAL_BATCH_SIZE})")
print("=" * 80)

# Run the training
if __name__ == "__main__":
    training_module.main()
