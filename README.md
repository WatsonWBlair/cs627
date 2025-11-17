# cs627

This repo is to coordinate work for Project 5 of CS627: Artificial Intelligence.

## SETUP
This project uses Meta's SONAR which relies on a MacOS (via a dependency on fairseq2)

Note that if you are using a system that fairseq2 does not provide a build for, you will need to compile it from source. See [this guide](https://github.com/facebookresearch/fairseq2/blob/main/INSTALL_FROM_SOURCE.md) for instructions on how to do this.

It is expected that you are using Conda to manage your local Python environment.

Execute the following command to ensure perquisites are installed:
```
sudo conda install -c conda-forge libsndfile sonar-space fairseq2
```


## Project Overview:
Full project specifications are located in the `documentation` directory.

The team has been examining the role of Semantics and Internal Latent Spaces in Natural/Spoken Language Understanding.

This project will examine the efficacy using models fine tuned to a shared semantic space for an NLP task. The overall architecure will rely on Encode/Decoder to translate text modalidy data into a modality models trained to the shared semantic space can consume and output. This transitions them from a text modality to a latent space modality.

We will then compare model performance between text modality and latent-space modality.
1. MiniGPT
2. o4GPT
3. Open-Weight Models
4. DeepSeek
5. Claude Sonnet

Key benchmarks we should seek to evaluate are:
    - Task completion and performance.
        - F1 score, accuracy, ect
    - Infrence latency
    - Model uncertanty
    - Token utilization / infrence cost.

##### Target Task
- [GLU Benchmark for General Language Understanding Evaluation](https://gluebenchmark.com/)
- [SuperGLU](https://super.gluebenchmark.com/)


### Foundational Research:
All relevant publications reviewed by the team are located in the `litrature` directory. 
Each paper is acompanied by a .txt file that contains the main takeaways and points of interest.

### Implementation:
All software and implementaiton assets are contained within the `src` directory.
- Fine Tuning to Latenet Space
    - Contrastive Learning
- Evaluation / Testesting
- Train Shared Space encoder/decoder
    - and training protocol for other modalities.

- Modality Encoder/Decoder: Bert/Bart [https://arxiv.org/abs/1910.03771] - start with BART text encoder/decoder.

- Contrast Learning Module: cosine distance is used to compute the similarity of speech-text embeddings using infoNCE loss
- Model Harness Module: maps encoders to fine-tuned models for evaluation.

##### System Design
TBD


## Next Steps
- Custom built model that is trained on latent space from the begenning, and does not include encoder/decoder within it's original architecure.
- Chain of thought evaluation of latent space modality.
- Diffrent modalities for encoding/decoding (audio, image)

#### Division Of Labor
Paper Authorship
- Xavier
- Kumar
#### Implementation
- Watson: Training Architecture and Infrastructure
- Ram: Encoder/Decoder
- Swatej: Data Wrangling and Preparation
