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

    def forward(self, input):
        pass

    def encode(self, text: str) -> list[int]:
        reconstructed = self.text2vec_model.predict(text, source_lang="eng_Latn")
        return reconstructed
    
    def batchEncode(self, inputs: list[str]):
        results = [self.encode(input) for input in inputs]
        return results

    def transform(self, text: str) -> list[int]:
        return self.encode(text)

