# 🤖 Local Production-Ready RAG Application

A private, production-ready Retrieval-Augmented Generation (RAG) system built from scratch. This application allows you to ingest local PDF documents, process them into a local vector database, and perform semantic searches to answer queries using a completely free, locally hosted LLM.

## 📁 Project Architecture

This codebase strictly follows a modular separation of concerns:

- `src/ingestion/` - Extracts text data from raw PDF documents.
- `src/chunking/` - Slices document text into manageable pieces with context overlap.
- `src/embeddings/` - Converts text slices into mathematical meaning using HuggingFace.
- `src/vectordb/` - Handles local vector storage and persistence via ChromaDB.
- `src/retrieval/` - Queries the database using semantic similarity search.
- `src/prompts/` - Houses isolated system prompt templates.
- `src/llm/` - Coordinates local model orchestration using Ollama (Llama 3.2).
- `src/api/` - Hosts high-performance API endpoints via FastAPI.
- `src/utils/` - Contains common helper scripts like YAML configurations.

---

## 🛠️ Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed and running locally.

### 2. Installation

Clone or navigate to the directory and activate your virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
