"""
Knowledge Base Ingestion Pipeline
=================================
Reads the agronomic_rules.json, generates embeddings for each rule,
and populates the FAISS Vector Store.
"""

import os
import sys
import json
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
sys.path.append(str(REPO_ROOT))

from backend.embeddings.local_embedder import LocalEmbedder
from backend.vectorstore.faiss_store import FaissVectorStore
from backend.models.domain import EmbeddingRecord, Chunk

KNOWLEDGE_FILE = REPO_ROOT / "knowledge_base" / "agronomic_rules.json"

def main():
    print(f"Loading agronomic rules from {KNOWLEDGE_FILE}")
    with open(KNOWLEDGE_FILE, "r") as f:
        rules = json.load(f)
        
    embedder = LocalEmbedder()
    # FAISS store automatically loads existing index if available, or creates a new one
    vector_store = FaissVectorStore()
    
    records = []
    
    print(f"Embedding {len(rules)} rule(s)...")
    for rule_name, rule_data in rules.items():
        # Text to embed
        text_payload = f"Crop Disease Rule: {rule_name}. {rule_data.get('logic_rule_text', '')}"
        
        # Embed the text payload
        vec = embedder.embed_text(text_payload)
        
        # FaissVectorStore mock chunk reconstruction expects 'text' in metadata
        meta = rule_data.copy()
        meta["text"] = text_payload
        
        records.append(EmbeddingRecord(
            chunk_id=uuid4(),
            embedding=vec,
            metadata=meta
        ))
        
    print("Populating FAISS index...")
    vector_store.add_embeddings(records)
    vector_store.save_index()
    
    print("FAISS index populated and saved successfully.")

if __name__ == "__main__":
    main()
