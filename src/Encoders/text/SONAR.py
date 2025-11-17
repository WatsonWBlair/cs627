import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SONAR():
    def __init__(self):
        self.text2vec_model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=DEVICE,
            dtype=torch.float16)


    def encode(self, semantic_vectors: list[str]) -> list[int]:
        reconstructed = self.text2vec_model.predict(semantic_vectors, source_lang="eng_Latn")
        return reconstructed