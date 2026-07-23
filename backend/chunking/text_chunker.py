from typing import List
import re
from backend.core.interfaces import BaseChunker
from backend.models.domain import Document, Chunk
from backend.core.exceptions import ChunkingError
from backend.config.settings import settings
from backend.logging.logger import chunk_logger

class RecursiveCharacterChunker(BaseChunker):
    """
    Recursively splits text using characters to respect max chunk size and overlap.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingError("chunk_overlap must be strictly less than chunk_size")

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        chunk_logger.info(f"Chunking {len(documents)} documents. size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        for doc in documents:
            try:
                text = doc.content
                # Simple naive chunking by character length for MVP
                # A full recursive chunker would split by \\n\\n, then \\n, then spaces.
                chunks_text = self._split_text(text)
                
                for idx, chunk_str in enumerate(chunks_text):
                    all_chunks.append(Chunk(
                        document_id=doc.id,
                        text=chunk_str.strip(),
                        chunk_index=idx,
                        metadata={"source": doc.source, **doc.metadata}
                    ))
            except Exception as e:
                chunk_logger.error(f"Error chunking document {doc.id}: {str(e)}")
                raise ChunkingError(f"Failed to chunk document {doc.id}: {str(e)}")
                
        chunk_logger.info(f"Generated {len(all_chunks)} chunks total.")
        return all_chunks
        
    def _split_text(self, text: str) -> List[str]:
        """Simple sliding window text splitter"""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If we're not at the end, try to find a nice breaking point (like a space)
            if end < text_len:
                # Find the last space within our window to avoid splitting words
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start + self.chunk_size // 2:
                    end = last_space
                    
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
            # Prevent infinite loop if overlap is weirdly handled
            if start >= end:
                start = end
                
        return chunks
