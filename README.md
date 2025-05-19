This is the general  overview of the project and technical documentation. For  a simpler overview see the [slides](https://github.com/rj-may/yoda_chat/blob/main/Presentation_Yoda.pptx)


# Yoda Trivia Chat LLM Project

## Overview

This project is a lightweight chatbot that uses a fine-tuned version of LLaMA 3.1 to respond to trivia-based queries in the unique, wise tone of Jedi Master Yoda. The system combines LoRA fine-tuning with retrieval-augmented generation (RAG), offering access through a Gradio UI and a FastAPI endpoint.

---

## 1. Model

### Base Model

* **Model Family**: LLaMA 3.1 (8B-Instruct, 4-bit quantized using Unsloth)
* **Context Window**: 2048 tokens
* **Temperature**: 0.7

### Fine-Tuning Details

* **Training Library**: Unsloth
* **Method**: LoRA (Low-Rank Adaptation)
* **Dataset**: Synthetic Yoda-style input/output pairs (\~80 examples)
* **Training Configuration**:

  * Max Steps: 18
  * Learning Rate: 2e-4
  * Batch Size: 1
  * No FP16 training

---

## 2. Inference Pipeline

1. User sends a natural language input.
2. Input is embedded using SentenceTransformer ("all-MiniLM-L6-v2").
3. FAISS index searches for the top-2 relevant trivia documents.
4. Retrieved context + user input are used to build a prompt.
5. Prompt is sent to the Yoda LLM via the locally hosted Ollama API.
6. Response is returned and formatted for display.

---

## 3. API and Applications

### Gradio App (app.py)

* **Purpose**: User-facing chat app
* **Features**:

  * Connects to local Ollama instance
  * Retrieves trivia from FAISS index
  * Dark-themed UI (custom CSS)
  * Remote sharing support

### FastAPI App (api\_app.py)

* **Purpose**: Programmatic API interface
* **Endpoint**: POST /yoda
* **Request**: JSON with 'user\_input' field
* **Response**: JSON with 'response' field
* **Status**: Working locally, remote access planned

---

## 4. Deployment Process

### Local Setup

1. Start FAISS vector DB and ensure embeddings are loaded.
2. Start Ollama server with Yoda LLM model.
3. Launch either:

   * Gradio app: `python app.py`
   * FastAPI app: `uvicorn api_app:app --reload`

### Future Enhancements

* Deploy FastAPI with `uvicorn + nginx`
* Enable Gradio remote sharing
* Use `llama.cpp` with merged GGUF model for portable inference

---

## 5. Sampling Method

### Decoding Strategy

* **Temperature**: 0.7
* **Sampling**: Top-p (nucleus sampling)

### Effect

* Encourages diverse, in-character Yoda phrasing
* Avoids repetition while maintaining coherence

---

## 6. Summary

The Yoda LLM project merges efficient LoRA fine-tuning with RAG to deliver creative and context-aware Yoda-style responses. The system offers:

* An interactive Gradio-based frontend
* A scalable FastAPI backend

---

## Appendix

### Prompt Engineering

Effective prompt used:

```
You are a wise Jedi master who speaks like Yoda. Use the trivia context to guide your answer.

Read this as your knowledge and trivia information:
{retrieved_context}

Read this as the USER prompt:
{user_input}

Answer:
```

### Lessons Learned

* Generic prompts without structured context often led to incorrect interpretations.
* Converting the model to GGUF was time-consuming:

  * `llama.cpp` didn’t work with Unsloth
  * Required Unsloth-based conversion
* Half of the project time was spent converting the model and handling text data.

---

> "Do or do not. There is no try." — Yoda
