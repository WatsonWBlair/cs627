import torch
import torch.nn as nn

# TODO: add support for saving and loading pretrained adaptors
class Adapter(nn.Module):
    config_class = 'NLP'
    def __init__(self, prefix: str, input_length = 1024, output_length = 1024, hidden_size = 200, hidden_layers = 2):
        super().__init__()
        self.weights_path = f"AdapterWeights/{prefix}_weights.pth"
        
        model_layers = [nn.Linear(input_length, hidden_size), nn.ReLU()]
        for i in range(hidden_layers):
            model_layers.append(nn.Linear(hidden_size, hidden_size))
            model_layers.append(nn.ReLU())
        model_layers.append(nn.Linear(hidden_size, output_length))

        self.model = nn.Sequential(*model_layers)

    def load(self):
        weights = torch.load(self.weights_path)
        self.model.load_state_dict(weights)

    def save(self):
        torch.save(self.model.state_dict(), self.weights_path)

    def forward(self, input_vectors, labels=None):
        return self.model(input_vectors)

        # Disabled for simplicity
        # loss = None
        # if labels is not None:
        #     # TODO: Custom Loss
        #     loss_fct = nn.CosineEmbeddingLoss()
        #     loss = loss_fct(output_vectors, labels)

        # return {"loss": loss, "output_vector": output_vectors}