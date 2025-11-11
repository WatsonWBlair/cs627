# cs627

This repo is to coordinate work for Project 5 of CS627: Artificial Intelegance.

### Project Overview:
Full project specifications are located in the `documentation` directory.

The team has been examining the role of Semantics and Internal Latent Spaces in Natural/Spoken Language Understanding.

This project will examine the efficacy of using a continuious latent space for a Multi-Agent AI System.

Key benchmarks we should seek to evaluate are:
- System Modularity:
    - How well do disprite models learn the shared latent space?
- System Performance:
    - Task completion and performance.
    - Overall system latency?
    - Overall system uncertanty?
    - Task-specific latency?
        - Internal Query Opperations (RAG on internal vector database)
        - MCP Tool-Use opperations
- Resources Utalization:
    - What is the impact on State/Context Size?
    - Number of reasoning steps required.

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