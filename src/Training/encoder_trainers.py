import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
from typing import Optional, Tuple

"""
This file contains any custom trainers for aligning Encoders to the shared semantic space.
"""

# TODO: Add class documentation
class Contrast(Trainer):
    def compute_loss(self, model: nn.Module, inputs: list[Tuple[str, torch.Tensor, torch.Tensor]], return_outputs: bool = False, num_items_in_batch: Optional[torch.Tensor] = None):
        margin = .1
        
        query = inputs[:][:1] # First item from every tuple
        positive = inputs[:][1:2] # Second item from every tuple
        negitive = inputs[:][-1] # Third item from every tuple
        outputs = model(*query)

        loss = []
        for q, p, n in zip(outputs, positive, negitive):
            maxMe = F.cosine_similarity(q,p)
            minMe = F.cosine_similarity(q,n)
            loss.append(max([maxMe - minMe + margin,0]))
       
        return (loss, outputs) if return_outputs else loss
    

# Add additional training paradimes below: