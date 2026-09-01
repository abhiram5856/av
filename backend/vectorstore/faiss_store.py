import os
import faiss
import numpy as np
from typing import List, Dict, Optional
from uuid import UUID

from backend.core.interfaces import BaseVectorStore
from backend.models.domain import EmbeddingRecord, SearchResult, Chunk
from backend.core.exceptions import VectorStoreError
from backend.config.settings import settings
from backend.core_logging.logger import retrieve_logger

class FaissVectorStore(BaseVectorStore):
    """
    Vector storage and fast retrieval using Facebook AI Similarity Search (FAISS).
    """

    def __init__(self, index_path: str = None, vector_dim: int = 384):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.vector_dim = vector_dim
        
        # We also need a way to map FAISS integer IDs back to our Chunk UUIDs/metadata
        self.doc_store: Dict[int, Chunk] = {}
        self._current_id = 0
        
        # Initialize or load FAISS Index
        # We use IndexFlatL2 for simple Euclidean distance, suitable for typical normalized embeddings
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # If user had a massive dataset, we'd use IVFPQ here instead of FlatL2.
        
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            retrieve_logger.info(f"Initialized empty FAISS index with dim={self.vector_dim}")

    def add_embeddings(self, records: List[EmbeddingRecord]) -> None:
        if not records:
            return
            
        retrieve_logger.info(f"Adding {len(records)} embeddings to FAISS store.")
        
        try:
            # Prepare vectors for FAISS (requires float32 numpy array)
            vectors = [record.embedding for record in records]
            np_vectors = np.array(vectors, dtype=np.float32)
            
            # Map the records
            for record in records:
                # In a real db we'd reconstruct the Chunk object or fetch it from a SQLite db.
                # Here we mock the chunk mapping for architectural completeness.
                mock_chunk = Chunk(
                    id=record.chunk_id,
                    document_id=record.id, # mock 
                    text=record.metadata.get("text", ""),
                    chunk_index=0,
                    metadata=record.metadata
                )
                self.doc_store[self._current_id] = mock_chunk
                self._current_id += 1
                
            self.index.add(np_vectors)
            retrieve_logger.debug(f"FAISS index total vectors: {self.index.ntotal}")
            
        except Exception as e:
            retrieve_logger.error(f"Failed to add embeddings: {str(e)}")
            raise VectorStoreError(f"FAISS insertion failed: {str(e)}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        if self.index.ntotal == 0:
            retrieve_logger.warning("Attempted to search an empty FAISS index.")
            return []
            
        try:
            query_vec = np.array([query_embedding], dtype=np.float32)
            
            # Perform search: distances and indices
            distances, indices = self.index.search(query_vec, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx in self.doc_store:
                    # Score represents distance (lower is better for L2)
                    chunk = self.doc_store[idx]
                    score = float(distances[0][i])
                    results.append(SearchResult(chunk=chunk, score=score))
                    
            return results
            
        except Exception as e:
            retrieve_logger.error(f"Search failed: {str(e)}")
            raise VectorStoreError(f"FAISS search failed: {str(e)}")
            
    def _load_index(self):
        try:
            self.index = faiss.read_index(self.index_path)
            doc_store_path = f"{self.index_path}.pkl"
            import pickle
            if os.path.exists(doc_store_path):
                with open(doc_store_path, "rb") as f:
                    data = pickle.load(f)
                    self.doc_store = data.get("doc_store", {})
                    self._current_id = data.get("current_id", 0)
            retrieve_logger.info(f"Loaded FAISS index from {self.index_path} with {self.index.ntotal} vectors.")
        except Exception as e:
            retrieve_logger.error(f"Failed to load FAISS index: {e}")
            self.index = faiss.IndexFlatL2(self.vector_dim)

    def save_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            doc_store_path = f"{self.index_path}.pkl"
            import pickle
            with open(doc_store_path, "wb") as f:
                pickle.dump({
                    "doc_store": self.doc_store,
                    "current_id": self._current_id
                }, f)
            retrieve_logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            retrieve_logger.error(f"Failed to save FAISS index: {e}")
