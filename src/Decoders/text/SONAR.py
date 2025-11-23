import torch
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SONAR():
    def __init__(self):
        self.vec2text_model = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=DEVICE,
            dtype=torch.float16)

    def decode(self, input_text: str):
        semantic_vectors = self.vec2text_model.predict(input_text, source_lang="eng_Latn")
        return semantic_vectors