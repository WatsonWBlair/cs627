# cs627

This repo is to coordinate work for Project 5 of CS627: Artificial Intelegance.

### Project Overview:
Full project specifications are located in the `documentation` directory.

The team has been examining the role of Semantics and Internal Latent Spaces in Natural/Spoken Language Understanding.

This project will examine the efficacy using models fine tuned to a shared semantic space for an NLP task. The overall architecure will rely on Encode/Decoder to translate text modalidy data into a modality models trained to the shared semantic space can consume and output.

We will then compare model performance between text modality and latent-space modality.
- MiniGPT
- Other Open-Weight Models

Key benchmarks we should seek to evaluate are:
    - Task completion and performance.
        - F1 score, accuracy, ect
    - Infrence latency
    - Model uncertanty
    - Token utilization / infrence cost.

##### Target Task
- Multi-Modal Sentement Analysis: Take audio and textual data from interviews and generate a sinopsis/sentement analysis of the content. eg: Celeberity Interview.
- 


### Foundational Research:
All relevant publications reviewed by the team are located in the `litrature` directory. 
Each paper is acompanied by a .txt file that contains the main takeaways and points of interest.

### Implementation:
All software and implementaiton assets are contained within the `src` directory.

##### System Design


## Division Of Labor
- Xavier: Paper-Writing Crew
- Watson: Implemetation Focused