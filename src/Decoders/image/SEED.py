import torch
import tensor
from diffusers import StableDiffusionPipeline
from torch import Generator

from utils.Adapter import Adapter
# load a fine-tuned image captioning model and corresponding tokenizer and image processor



BASE_MODEL = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SEED():
    def __init__(self, use_saved: bool, base_model: str = BASE_MODEL, max_length: int = 1024) -> None:
        super(SEED, self).__init__()

        self.generator = Generator(device=DEVICE)
        self.safe_pipeline = StableDiffusionPipeline.from_pretrained(base_model, safety_checker=None)
        self.safe_pipeline.text_encoder = Adapter(max_length,200,max_length)
       
    def forward(self, semantic_vector: tensor.Tensor, device: str = DEVICE):
        output = self.safe_pipeline.to(device)(prompt=semantic_vector, generator=self.generator).images[0]
        return output

