import pytest
from backend.chunking.text_chunker import RecursiveCharacterChunker
from backend.models.domain import Document
from backend.core.exceptions import ChunkingError

def test_chunker_initialization():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 20

def test_chunker_invalid_overlap():
    with pytest.raises(ChunkingError, match="strictly less than"):
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=50)

def test_chunk_split_logic():
    chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=2)
    doc = Document(content="0123456789abcdefghij", source="test.txt")
    
    # 0123456789 -> length 10
    # overlap 2 -> back to index 8. Next string starts at 8: 89abcdefgh -> length 10
    # overlap 2 -> back to index 16. Next string starts at 16: ghij -> length 4
    
    chunks = chunker.chunk([doc])
    
    assert len(chunks) == 3
    assert chunks[0].text == "0123456789"
    assert chunks[1].text == "89abcdefgh"
    assert chunks[2].text == "ghij"
    
    assert chunks[0].document_id == doc.id
    assert chunks[0].metadata["source"] == "test.txt"

def test_word_boundary_splitting():
    # Chunker should try to find spaces
    chunker = RecursiveCharacterChunker(chunk_size=15, chunk_overlap=2)
    doc = Document(content="hello world this is a test", source="test.txt")
    
    chunks = chunker.chunk([doc])
    # The chunker logic finds the last space before end.
    assert len(chunks) > 0
    # Ensures no chunk is larger than 15 chars
    for chunk in chunks:
        assert len(chunk.text) <= 15
