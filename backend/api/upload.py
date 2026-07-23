import os
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from backend.document_loader.pdf_loader import PDFLoader
from backend.chunking.text_splitter import TextSplitter
from backend.embeddings.local_embedder import LocalEmbedder
from backend.vectorstore.faiss_store import FaissVectorStore
from backend.models.domain import EmbeddingRecord

router = APIRouter()

# Initialize Pipeline Components (Singleton pattern for the endpoint)
# Note: In a production app, these would be injected as dependencies
splitter = TextSplitter(chunk_size=800, overlap=100)
embedder = LocalEmbedder(model_name='all-MiniLM-L6-v2')
vector_store = FaissVectorStore()

@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF) to be ingested into the knowledge base (RAG).
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported at this time.")
        
    try:
        # 1. Extract Text
        document_id = str(uuid4())
        raw_text = await PDFLoader.extract_text(file)
        
        if not raw_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. It might be scanned or empty.")
            
        # 2. Chunk Text
        chunks = splitter.split_text(raw_text, document_id=document_id, document_name=file.filename)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from the document.")
            
        # 3. Generate Embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_batch(chunk_texts)
        
        # 4. Save to FAISS
        records = []
        for i, chunk in enumerate(chunks):
            records.append(
                EmbeddingRecord(
                    id=document_id,
                    chunk_id=chunk.id,
                    embedding=embeddings[i],
                    metadata=chunk.metadata
                )
            )
            
        vector_store.add_embeddings(records)
        vector_store.save_index()
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "document_id": document_id,
            "chunks_processed": len(chunks),
            "total_vectors_in_store": vector_store.index.ntotal
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
