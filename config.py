import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file if it exists (for local development)
if os.path.exists(".env"):
    load_dotenv()

# Try loading from Streamlit secrets first, fallback to environment variables
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY"))
ARXIV_EMAIL = st.secrets.get("ARXIV_EMAIL", os.getenv("ARXIV_EMAIL"))

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  


GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  
# GENERATION_MODEL = "google/gemma-7b-it"  
# GENERATION_MODEL = "meta-llama/Llama-2-13b-chat-hf"  
# GENERATION_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  
QUERY_EXPANSION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  

DIMENSION = 768 


ENABLE_QUERY_EXPANSION = True  


SEARCH_RELEVANCE_THRESHOLD = 1.2  
AUTO_RESEARCH = True  


MODEL_CONFIG = {
    "max_new_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
  
    "context_length": 8192,  
    "streaming": False,
    "return_full_text": False
}


INDEX_PATH = "data/faiss_index"


CACHE_DIR = "data"
FAISS_INDEX_DIR = "data/faiss_index"
FAISS_INDEX_PATH = f"{FAISS_INDEX_DIR}/index.faiss"
FAISS_METADATA_PATH = f"{FAISS_INDEX_DIR}/metadata.pkl"
ARXIV_CACHE_PATH = "data/cache/arxiv_cache.json"
WIKI_CACHE_PATH = "data/cache/wiki_cache.json"
LOG_PATH = "data/logs/retrieval.log"


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"


def ensure_directories():
    """Create all necessary directories for data storage"""
    directories = [
        CACHE_DIR,
        FAISS_INDEX_DIR,
        os.path.dirname(ARXIV_CACHE_PATH),
        os.path.dirname(WIKI_CACHE_PATH),
        os.path.dirname(LOG_PATH)
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
