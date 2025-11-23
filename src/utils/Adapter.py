import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers = 2, pre: str = "facebook/bart-base"):
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.hidden_layers = [nn.ReLU()]
        for i in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.input_layer(x)
        
        for layer in self.hidden_layers:
            out = layer(out)
        
        result = self.output_layer(out)
        return result