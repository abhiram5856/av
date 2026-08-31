import pytest
import numpy as np
from uuid import uuid4
import tempfile
import os
from backend.vectorstore.faiss_store import FaissVectorStore
from backend.models.domain import EmbeddingRecord

def test_faiss_store_initialization():
    # Use a temp path that doesn't exist so the store starts empty
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_path = os.path.join(tmpdir, "test_index")
        store = FaissVectorStore(index_path=empty_path, vector_dim=3)
        assert store.vector_dim == 3
        assert store.index.ntotal == 0

def test_faiss_add_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_path = os.path.join(tmpdir, "test_index")
        store = FaissVectorStore(index_path=empty_path, vector_dim=2)
        
        # Create mock embedding records
        rec1 = EmbeddingRecord(chunk_id=uuid4(), embedding=[1.0, 0.0], metadata={"text": "Crop disease"})
        rec2 = EmbeddingRecord(chunk_id=uuid4(), embedding=[0.0, 1.0], metadata={"text": "Weather info"})
        
        # Add to store
        store.add_embeddings([rec1, rec2])
        
        assert store.index.ntotal == 2
        assert len(store.doc_store) == 2
        
        # Search for something close to rec1
        results = store.search([0.9, 0.1], top_k=1)
        
        assert len(results) == 1
        assert results[0].chunk.text == "Crop disease"
        # Ensure score exists
        assert results[0].score is not None

def test_faiss_search_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_path = os.path.join(tmpdir, "test_index")
        store = FaissVectorStore(index_path=empty_path, vector_dim=2)
        results = store.search([1.0, 0.0])
        assert len(results) == 0
