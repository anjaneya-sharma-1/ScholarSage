import faiss
import numpy as np
import pickle
import os
import logging
import config

logger = logging.getLogger('retrieval')

class FAISSIndex:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.chunks = []
        self.chunk_indices = {}  
        
    def create_new_index(self, dimension=config.DIMENSION):
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.chunks = []
        self.chunk_indices = {}
        
    def add_to_index(self, embeddings, metadata_list, chunks=None):
        if len(embeddings) > 0:
            start_idx = self.index.ntotal
            self.index.add(np.array(embeddings))
            
        
            if chunks is None:
                chunks = []
                for meta in metadata_list:
                    if 'chunk_text' in meta:
                        chunks.append(meta.pop('chunk_text'))
                    else:
                        chunks.append("")
                        
            # Store chunks and their metadata
            for i, (meta, chunk) in enumerate(zip(metadata_list, chunks)):
                chunk_id = start_idx + i
                self.metadata.append(meta)
                self.chunks.append(chunk)
                self.chunk_indices[chunk_id] = len(self.chunks) - 1
            
    def search(self, query_vector, k=5):
        if self.index is None:
            return None, None
        return self.index.search(query_vector, k)
            
    def save(self):
        os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, config.FAISS_INDEX_PATH)
        with open(config.FAISS_METADATA_PATH, 'wb') as f:  # Fixed typo in variable name
            pickle.dump({
                'metadata': self.metadata,
                'chunks': self.chunks,
                'chunk_indices': self.chunk_indices
            }, f)
        logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
            
    def load(self):
        if os.path.exists(config.FAISS_INDEX_PATH) and os.path.exists(config.FAISS_METADATA_PATH):
            try:
                self.index = faiss.read_index(config.FAISS_INDEX_PATH)
                with open(config.FAISS_METADATA_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get('metadata', [])
                    self.chunks = data.get('chunks', [])
                    self.chunk_indices = data.get('chunk_indices', {})
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                return True
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
                return False
        return False

# These are standalone helper functions for backward compatibility
def create_index(dimension=config.DIMENSION):
    """Create a new FAISS index with the specified dimension"""
    return faiss.IndexFlatL2(dimension)

def add_to_index(index, embeddings):
    """Add embeddings to the index"""
    if len(embeddings) > 0:
        index.add(np.array(embeddings))

def search_index(index, query_vector, k=5):
    """Search the index for the k nearest neighbors to the query vector"""
    return index.search(query_vector, k)
