from typing import List
from uuid import uuid4
from backend.models.domain import Chunk

class TextSplitter:
    """
    Splits long document text into smaller overlapping chunks for optimal vector search.
    """
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str, document_id: str, document_name: str) -> List[Chunk]:
        """
        Splits a string into overlapping chunks and returns a list of Chunk objects.
        """
        chunks = []
        start = 0
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            # End index is start + chunk_size
            end = start + self.chunk_size
            
            # If we're not at the end of the text, try to find a nice breaking point (like a newline or period)
            if end < text_length:
                # Look back for a period to avoid cutting sentences in half
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + (self.chunk_size / 2):
                    end = last_period + 1
                    
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > 20: # Ignore tiny useless chunks
                chunks.append(
                    Chunk(
                        id=uuid4(),
                        document_id=document_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        metadata={"source": document_name}
                    )
                )
                chunk_index += 1
                
            # Move the start forward, keeping the overlap
            start = end - self.overlap

        return chunks
