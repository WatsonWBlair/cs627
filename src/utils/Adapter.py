import torch
import torch.nn as nn

from transformers import PreTrainedModel

# TODO: add support for saving and loading pretrained adaptors
class Adapter(nn.Module):
    config_class = 'NLP'
    def __init__(self, token_length = 1024, hidden_size = 200, hidden_layers = 2):
        """
        Config options available https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/configuration#transformers.PretrainedConfig
        """
        super().__init__()
        self.input_layer = nn.Linear(token_length, hidden_size)
        
        self.hidden_layers = [nn.ReLU()]
        for i in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(hidden_size, token_length)

    def forward(self, input_vectors, labels=None):
        out = self.input_layer(input_vectors)
        
        for layer in self.hidden_layers:
            out = layer(out)
        
        output_vectors = self.output_layer(out)

        loss = None
        if labels is not None:
            loss_fct = nn.CosineEmbeddingLoss()
            loss = loss_fct(output_vectors, labels)
        return {"loss": loss, "output_vector": output_vectors}