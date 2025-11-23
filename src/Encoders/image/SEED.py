import requests
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import logging
import tensor
from transformers import  ViTImageProcessor, VisionEncoderDecoderModel

from utils.Adapter import Adapter
# load a fine-tuned image captioning model and corresponding tokenizer and image processor


BASE_MODEL = "nlpconnect/vit-gpt2-image-captioning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SEED():
    def __init__(self, use_saved: bool, base_model: str = BASE_MODEL, max_length: int = 1024) -> None:
        super(SEED, self).__init__()
        adapterModel = Adapter(max_length,200,max_length) # TODO: Load from checkpoint using pretrained arg
        self.model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL, adapterModel)
        self.tokenizer = ViTImageProcessor.from_pretrained(BASE_MODEL) #image_processor

       
    def forward(self, input_img: tensor.Tensor, *, max_length: int = 1024, device: str = DEVICE):
        pixel_values = self.tokenizer(input_img, return_tensors="pt", max_length=max_length).to(device).pixel_values
        output = self.model.generate(pixel_values, skip_special_tokens=True)[0]

        return output


