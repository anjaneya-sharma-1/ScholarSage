# ScholarSage 📚

**An intelligent research assistant powered by state-of-the-art language models, vector embeddings, and semantic search.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32.0-red.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-vectordb-orange.svg)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/Hugging-Face-yellow.svg)](https://huggingface.co/)


![image](https://github.com/user-attachments/assets/e206a77e-a259-4b94-a251-7d9f447372c0)

## Overview

ScholarSage is an AI-powered research assistant that leverages vector embeddings, semantic search, and large language models to help researchers explore and analyze academic content. The system automatically fetches information from Wikipedia and ArXiv, processes PDF documents, and provides intelligent answers to research questions through a conversational interface.

![ScholarSage Interface](https://example.com/scholarsage-screenshot.png)

## Features

- **Intelligent Semantic Search**: Uses advanced vector embeddings to understand the meaning of your questions and find relevant information
- **Automatic Web Research**: Proactively fetches information from Wikipedia and ArXiv when asked about new topics
- **PDF Document Processing**: Upload and extract knowledge from research papers and documents
- **Query Expansion**: Automatically reformulates queries to improve search effectiveness
- **Multiple Knowledge Sources**: Integrates Wikipedia articles, ArXiv papers, and uploaded PDFs
- **Conversational Interface**: Clean, modern chat interface for natural interaction
- **Flexible Model Selection**: Choose between different LLMs (Mistral, Gemma, Mixtral, Llama) based on your needs
- **Fallback Mechanisms**: Multiple backup strategies to ensure reliable answers even when primary methods fail

## Architecture

ScholarSage is built with a modular architecture:

1. **Frontend Layer**: Streamlit-based UI providing a chat interface and configuration options
2. **Vector Search Layer**: FAISS-powered vector database for efficient semantic search
3. **Knowledge Acquisition Layer**: Components for retrieving data from Wikipedia, ArXiv, and PDFs
4. **Embedding Layer**: Sentence-transformers for converting text to vector embeddings
5. **Generation Layer**: HuggingFace models for answer generation and query expansion
6. **Orchestration Layer**: Logic for coordinating the flow between components

## Installation

### Prerequisites

- Python 3.8+
- Git
- Hugging Face account with API key

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/scholarsage.git
cd scholarsage

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

### Environment Setup

1. After copying the example environment file, edit `.env` with your API keys:
   ```
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   ARXIV_EMAIL=your_email_for_arxiv_api@example.com
   ```

2. These keys will be automatically loaded by the application and kept out of version control for security.

## Directory Structure

```
ScholarSage/
│── app.py                      # Main Streamlit app
│── requirements.txt            # Required dependencies
│── config.py                   # Configurations (API keys, etc.)
│── data/                       # Directory for cached data
│── utils/                      # Helper functions
│   ├── faiss_helper.py         # FAISS index helper functions
│   ├── fetch_data.py           # Data fetching functions
│   ├── embeddings.py           # Embedding model integration
│── models/                     # Store pre-trained models
```
