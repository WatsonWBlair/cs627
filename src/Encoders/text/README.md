# SONAR Latent Space Simulation (BART)

This repository contains a conceptual Python simulation of the core mechanics behind systems like **SONAR** (Sentence-level multimOdal and laNguage-Agnostic Representations). These systems aim to map diverse inputs—such as text, speech, and multiple languages—into a unified, fixed-size numerical space known as the **Latent Space**.

This simulation uses the Hugging Face **BART-base** model (`facebook/bart-base`) to demonstrate three core components of the architecture.

---

## 🚀 Key Concepts

### 1. **The Shared Latent Space**

The **Latent Space** is the unified semantic space where all modalities and languages are mapped.

- **Goal:** Represent the meaning of any input (e.g., English text, French audio, etc.) as a single vector.  
- **Vector Size:** In this BART simulation, the size is **768 dimensions**.

---

### 2. **Encoder Function: Text → Latent Vector (t2vec)**

The function `encode_text_to_latent` converts text into its latent-space representation.

**How it works:**

1. Input text is fed into the **BART encoder**.
2. The encoder returns a sequence of hidden-state vectors.
3. These vectors are **mean-pooled** (averaged) across the sequence.
4. The result is a single **fixed-size semantic vector**.

This vector captures the **semantic fingerprint** of the sentence.

---

### 3. **Decoder Function: Latent Vector → Text (vec2text)**

The function `decode_latent_to_text` simulates reconstructing text from a latent vector.

#### ⚠️ The Challenge
Models like BART **cannot** directly decode from a single vector, since they expect a full sequence of encoder hidden states for cross-attention.

#### ✅ The Simulation
For demonstration purposes, the simulation re-runs normal BART text generation using its internal encoder, conceptually representing Vec → Text behavior.

This allows the script to show the full **Text → Latent → Text** pipeline.

---

## ⚙️ Requirements

Install the required libraries:

```bash
pip install torch transformers numpy
