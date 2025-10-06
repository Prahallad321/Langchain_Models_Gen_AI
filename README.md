# LangChain Models Documentation

## Overview
This project demonstrates how to integrate and use different **Large Language Models (LLMs)** and **Embedding Models** within the **LangChain** framework.  
It covers multiple APIs and local model setups, including OpenAI, Anthropic, Google AI, and Hugging Face.

---

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [LLMs Overview](#llms-overview)
4. [Chat Models](#chat-models)
   - [OpenAI API](#openai-api)
   - [Anthropic API](#anthropic-api)
   - [Google AI API](#google-ai-api)
   - [Hugging Face API](#hugging-face-api)
   - [Hugging Face Local](#hugging-face-local)
5. [Embedding Models](#embedding-models)
   - [OpenAI Embeddings](#openai-embeddings)
   - [Hugging Face Local Embeddings](#hugging-face-local-embeddings)
6. [Environment Setup](#environment-setup)
7. [Configuration](#configuration)
8. [Key Notes](#key-notes)
9. [References](#references)

---

## 🧠 Introduction
The LangChain framework enables seamless interaction between different LLMs and external tools.  
This project focuses on exploring **multiple language model providers** and how they can be integrated into applications for text generation, embeddings, and conversational AI.

---

## 📁 Project Structure
Langchain_Models/
│
├── 1.LLMs/
│ ├── OpenAI/
│ ├── Anthropic/
│ ├── GoogleAI/
│ └── HuggingFace/
│
├── 2.ChatModels/
│ ├── chatmodel_openai.py
│ ├── chatmodel_anthropic.py
│ ├── chatmodel_google.py
│ ├── chatmodel_hf_api.py
│ └── chatmodel_hf_local.py
│
├── 3.EmbeddingModels/
│ ├── embedding_openai.py
│ └── embedding_hf_local.py
│
└── README.md


---

## ⚙️ LLMs Overview
LLMs (Large Language Models) are at the core of LangChain integrations.  
They provide the reasoning, summarization, and generative capabilities needed for natural language applications.

Each provider (OpenAI, Anthropic, Google AI, Hugging Face) offers its own API and configuration methods.  
This project demonstrates how to configure and use them interchangeably with LangChain’s unified API.

---

## 💬 Chat Models

### 1. OpenAI API
**Model Examples:**  
- `gpt-4`, `gpt-3.5-turbo`, `gpt-4o-mini`  
**Usage:**  
Used for text generation, summarization, Q&A, and multi-turn conversations.  
Requires an **OpenAI API key** set in the environment file.

**Environment Variable:**  


---

### 2. Anthropic API
**Model Examples:**  
- `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`  
**Usage:**  
Offers high-quality, instruction-following text generation and reasoning.  
Requires an **Anthropic API key**.

**Environment Variable:**  


---

### 3. Google AI API
**Model Examples:**  
- `gemini-1.5-pro`, `gemini-2.0-flash`  
**Usage:**  
Supports fast, efficient chat-based interactions with Google’s Gemini models.  
Requires a **Google API key**.

**Environment Variable:**  


---

### 4. Hugging Face API
**Model Examples:**  
- `mistralai/Mistral-7B-Instruct-v0.3`  
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Usage:**  
Run text generation or chat models via the Hugging Face Inference API.  
Requires a **Hugging Face access token**.

**Environment Variable:**  

---

### 5. Hugging Face Local
**Model Examples:**  
- `mistral-7b-instruct`, `falcon-7b`, `llama-2`  
**Usage:**  
Runs models locally without external API calls, ideal for offline or privacy-focused applications.  
Ensure you have PyTorch or TensorFlow installed for model loading.

---

## 🔍 Embedding Models
Embeddings convert text into numerical vectors for use in similarity search, clustering, and retrieval-augmented generation (RAG).

### 1. OpenAI Embeddings
**Model:**  
`text-embedding-3-large` or `text-embedding-3-small`  
Used for semantic search and vector database storage.

**Environment Variable:**  


---

### 2. Hugging Face Local Embeddings
**Model Examples:**  
- `sentence-transformers/all-MiniLM-L6-v2`  
- `BAAI/bge-small-en`  
**Usage:**  
Creates local embeddings without API usage.  
Useful for cost-free, offline operations.

---

## ⚙️ Environment Setup

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)
- Installed dependencies:


### Create `.env` file


These keys are required for accessing the respective APIs.

---

## 💬 Models Overview

### Chat Models

| Provider | Script | Description |
|-----------|---------|-------------|
| **OpenAI** | `1_chatmodel_openai.py` | Uses GPT-based chat models via OpenAI API. |
| **Anthropic** | `2_chatmodel_anthropic.py` | Integrates Claude models from Anthropic API. |
| **Google AI (Gemini)** | `3_chatmodel_google.py` | Uses Gemini models from Google Generative AI. |
| **Hugging Face API** | `4_chatmodel_hf_api.py` | Connects to models hosted on Hugging Face Hub using API key. |
| **Hugging Face Local** | `5_chatmodel_hf_local.py` | Runs models locally using `transformers` and LangChain. |

Each script demonstrates how to initialize, interact with, and generate responses from models using LangChain’s standardized interfaces.

---

### Embedding Models

| Provider | Script | Description |
|-----------|---------|-------------|
| **OpenAI Embeddings** | `1_openai_embedding.py` | Generates text embeddings using OpenAI’s embedding models such as `text-embedding-3-large`. |
| **Hugging Face Local Embeddings** | `2_huggingface_local_embedding.py` | Uses local models like `sentence-transformers/all-MiniLM-L6-v2` for offline embedding generation. |

Embeddings are vector representations of text that can be used for semantic similarity, clustering, search, and recommendation systems.

---


2. Choose a local model from Hugging Face:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `distilbert-base-uncased`
- `sentence-transformers/all-MiniLM-L6-v2`

3. Ensure your system meets minimum requirements:
- At least **8GB RAM**
- **GPU support** is recommended for larger models.

4. Run the respective local script to start inference.

---

## 📦 Dependencies

Main dependencies used in this project:


## 🧠 Local Hugging Face Setup

Running models locally allows you to use open-source models without API costs.  
Follow these steps to set it up:

1. Install required packages:



---

## 🧩 Configuration
- All environment variables are loaded using `python-dotenv`.
- LangChain automatically detects which API client to use.
- Local Hugging Face models require hardware acceleration (GPU recommended).

---

## 💡 Key Notes
- Each API may have different rate limits and token policies.
- For production use, cache responses and manage API retries.
- Use local Hugging Face models if privacy or offline capability is required.
- Embedding models can be combined with vector stores (like FAISS, Chroma, Pinecone).

---

## 📘 References
- [LangChain Documentation](https://python.langchain.com)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Google AI Gemini Docs](https://ai.google.dev/)
- [Hugging Face Docs](https://huggingface.co/docs)

---

## 🧾 License
This project is open for educational and research use.  
All API keys and credentials must be securely managed and not shared publicly.

---
